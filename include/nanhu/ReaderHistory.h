// Compare the exact key produced by stable false/true membership sorts in
// ascending reader order. Reader IDs must be sorted and unique.
#pragma once
#include <algorithm>

template <typename HistoryA, typename HistoryB>
bool readerHistoryLess(const HistoryA &a, const HistoryB &b) {
  auto ai = a.rbegin();
  auto bi = b.rbegin();
  while (ai != a.rend() && bi != b.rend()) {
    if (*ai != *bi) return *ai < *bi;
    ++ai;
    ++bi;
  }
  return ai == a.rend() && bi != b.rend();
}

template <typename Values, typename HistoryOf>
void sortByReaderHistory(Values &values, HistoryOf historyOf) {
  std::stable_sort(values.begin(), values.end(), [&](const auto &a, const auto &b) {
    return readerHistoryLess(historyOf(a), historyOf(b));
  });
}
