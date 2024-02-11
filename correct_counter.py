import torch as t


class Counter:

  def __init__(self, label: int):
    self.label = label
    self.n_correct = 0
    self.n_incorrect = 0

  def to_dict(self):
    return {
      "n_correct": self.n_correct,
      "n_incorrect": self.n_incorrect
    }

  def inc_correct(self, v: int):
    self.n_correct += v

  def inc_incorrect(self, v: int):
    self.n_incorrect += v


class CounterCollection:

  def __init__(self):
    self.counters: dict[int, Counter] = {}

  def __str__(self):
    out_d = {key: value.to_dict() for key, value in self.counters.items()}
    return str(out_d)

  def update_counter_unsafe(self, label: int, v: int, is_corrects: bool):
    if is_corrects:
      self.counters[label].inc_correct(v)
    else:
      self.counters[label].inc_incorrect(v)

  def upsert_counter(self, label: int, v: int, is_corrects: bool):
    counter_or_none = self.counters.get(label)
    if counter_or_none is None:
      self.counters[label] = Counter(label)
    self.update_counter_unsafe(label, v, is_corrects)

  def update(self, input, preds):
    for label_t in t.unique(t.cat((preds, input))):
      label = label_t.item()
      corrects_of_label = t.sum((preds == label) & (input == label)).item()
      self.upsert_counter(label, corrects_of_label, is_corrects=True)
      incorrects_of_label = t.sum((preds != label) & (input == label)).item()
      self.upsert_counter(label, incorrects_of_label, is_corrects=False)

