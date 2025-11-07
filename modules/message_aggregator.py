from collections import defaultdict
import torch
import numpy as np


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""    
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    
    to_update_node_ids = []
    
    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])
    
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class TemporalWeightedMeanAggregator(MessageAggregator):
  def __init__(self, device, beta=0.8):
    """
    :param device: The device to run the computations on.
    :param beta: A hyperparameter controlling the decay rate.
                 A larger beta means faster decay (only very recent messages matter).
                 A smaller beta means slower decay (past messages are more influential).
    """
    super(TemporalWeightedMeanAggregator, self).__init__(device)
    self.beta = beta

  def aggregate(self, node_ids, messages):
    """
    Aggregates messages using a temporal-weighted mean.
    :param node_ids: A list of all node ids (with duplicates)
    :param messages: A defaultdict[node_id, list[(message, timestamp)]]
    """
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    to_update_node_ids = []

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        to_update_node_ids.append(node_id)
        
        # 1. Get messages, timestamps, and find the last timestamp
        msg_list = messages[node_id]
        message_tensors = torch.stack([m[0] for m in msg_list])
        timestamp_tensors = torch.stack([m[1] for m in msg_list])
        last_timestamp = timestamp_tensors.max()
        unique_timestamps.append(last_timestamp)
        
        # 2. Calculate time lags and weights
        # time_lag = last_timestamp - timestamp_tensors
        # weights = torch.exp(-self.beta * time_lag)
        
        # Numerically stable version:
        # exp( -B * (T_last - t_i) ) = exp( -B*T_last ) * exp( B*t_i )
        # We can normalize out the exp(-B*T_last) term
        rel_timestamps = self.beta * (timestamp_tensors - last_timestamp)
        weights = torch.exp(rel_timestamps) # All <= 1.0
        weights = weights.unsqueeze(1) # Shape [n_messages, 1]
        
        # 3. Calculate weighted average
        weighted_sum = torch.sum(message_tensors * weights, dim=0)
        total_weight = torch.sum(weights)
        
        # Avoid division by zero if weights are tiny
        if total_weight == 0:
          agg_message = torch.mean(message_tensors, dim=0)
        else:
          agg_message = weighted_sum / total_weight
          
        unique_messages.append(agg_message)

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps

class LSTMAggregator(MessageAggregator):
  def __init__(self, device, message_dim, hidden_dim):
    """
    :param device: The device to run the computations on.
    :param message_dim: The dimension of the input messages.
    :param hidden_dim: The dimension of the LSTM's hidden state (output dimension).
    """
    super(LSTMAggregator, self).__init__(device)
    self.message_dim = message_dim
    self.hidden_dim = hidden_dim
    self.lstm_cell = torch.nn.LSTMCell(message_dim, hidden_dim).to(device)

  def aggregate(self, node_ids, messages):
    """
    Aggregates messages using an LSTM.
    :param node_ids: A list of all node ids (with duplicates)
    :param messages: A defaultdict[node_id, list[(message, timestamp)]]
    """
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    to_update_node_ids = []

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        to_update_node_ids.append(node_id)
        
        # 1. Sort messages by timestamp
        msg_list = messages[node_id]
        sorted_list = sorted(msg_list, key=lambda x: x[1])
        
        # 2. Initialize hidden and cell states
        # We process one node at a time, so batch size is 1
        h = torch.zeros(1, self.hidden_dim, device=self.device)
        c = torch.zeros(1, self.hidden_dim, device=self.device)
        
        # 3. Iterate through sorted messages
        for msg, timestamp in sorted_list:
          # LSTMCell expects input shape [batch_size, input_dim]
          # Our msg is [input_dim], so we unsqueeze it
          current_msg = msg.unsqueeze(0)
          h, c = self.lstm_cell(current_msg, (h, c))
          
        # 4. The final hidden state is the aggregated message
        # Squeeze to remove the batch dimension
        agg_message = h.squeeze(0)
        unique_messages.append(agg_message)
        
        # 5. Keep the timestamp of the last message
        unique_timestamps.append(sorted_list[-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps

import torch.nn.functional as F

class GATAggregator(MessageAggregator):
  def __init__(self, device, message_dim):
    """
    :param device: The device to run the computations on.
    :param message_dim: The dimension of the input messages.
    """
    super(GATAggregator, self).__init__(device)
    self.message_dim = message_dim
    
    # Learnable parameters for attention
    # 1. A linear transformation for the messages
    self.feat_transform = torch.nn.Linear(message_dim, message_dim, bias=False).to(device)
    
    # 2. The attention context vector (aT in the GAT paper)
    # We use a simplified dot-product attention
    self.attn_vec = torch.nn.Parameter(torch.empty(size=(message_dim, 1), device=device))
    torch.nn.init.xavier_uniform_(self.attn_vec.data, gain=1.414)
    
    self.leaky_relu = torch.nn.LeakyReLU(0.2)

  def aggregate(self, node_ids, messages):
    """
    Aggregates messages using a GAT-style attention mechanism.
    :param node_ids: A list of all node ids (with duplicates)
    :param messages: A defaultdict[node_id, list[(message, timestamp)]]
    """
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    to_update_node_ids = []

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        to_update_node_ids.append(node_id)
        
        # 1. Stack all messages for this node
        msg_list = messages[node_id]
        message_tensors = torch.stack([m[0] for m in msg_list]) # Shape [n_messages, msg_dim]
        timestamp_tensors = torch.stack([m[1] for m in msg_list])
        
        # 2. Transform features
        h = self.feat_transform(message_tensors) # Shape [n_messages, msg_dim]
        
        # 3. Calculate attention scores
        # (h @ self.attn_vec) gives shape [n_messages, 1]
        attn_scores = self.leaky_relu(torch.matmul(h, self.attn_vec)).squeeze(1) # Shape [n_messages]
        
        # 4. Normalize weights with softmax
        attn_weights = F.softmax(attn_scores, dim=0) # Shape [n_messages]
        
        # 5. Calculate weighted sum
        # We weight the *original* messages, not the transformed ones
        # attn_weights.unsqueeze(1) gives shape [n_messages, 1]
        agg_message = torch.sum(message_tensors * attn_weights.unsqueeze(1), dim=0)
        unique_messages.append(agg_message)
        
        # 6. Keep the timestamp of the last message
        unique_timestamps.append(timestamp_tensors.max())

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps

def get_message_aggregator(aggregator_type,message_dim, device):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device)
  elif aggregator_type == "temporal_weighted_mean":
    return TemporalWeightedMeanAggregator(device=device, beta=1.0)
  elif aggregator_type == "lstm":
    return LSTMAggregator(device=device, message_dim=128, hidden_dim=128)
  elif aggregator_type == "gat":
    return GATAggregator(device=device, message_dim=128)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))


