template <typename T>
void remove(std::size_t n, T* array, std::size_t m, const T* removed) {
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t j;
    auto x = array[i];
    for (j = 0; j < m; ++j) {
      if (removed[j] > x) {
        break;
      }
    }
    array[i] -= j;
  }
}
