#include <cstdint>
#include <iostream>

#include "uniquelist/unique_list.h"

using I = int32_t;
using T = double;

using unique_array_list =
    uniquelist::unique_array_list<T, uniquelist::strictly_less>;

extern "C" {

auto uniquelist_batched_unique_array_list_create(I batch_size, I array_size) {
  auto size = batch_size * sizeof(unique_array_list);
  auto *list = static_cast<unique_array_list *>(operator new[](size));
  for (I i = 0; i < batch_size; ++i) {
    new (&list[i]) unique_array_list(array_size);
  }
  return list;
}

auto uniquelist_batched_unique_array_list_delete(unique_array_list *list,
                                               I batch_size) {
  for (I i = batch_size - 1; i >= 0; --i) {
    list[i].~unique_array_list();
  }
  operator delete[](list);
}

auto uniquelist_batched_unique_array_list_size(unique_array_list *list, I batch_size,
                                             I *list_size) {
  for (I i = 0; i < batch_size; ++i) {
    list_size[i] = list[i].size();
  }
}

/**
 * @brief Append items to the end
 *
 * This appends multiple items to the end.
 *
 * @param [in] list Pointer to the array of uniquelist.
 * @param [in] array_size Size of arrays in the lists.
 * @param [in] n_items Number of items to be added.
 * @param [in] list_index List indexes on which corresponding items
 *     are inserted.  size: n_items
 * @param [in] value A concatenation of values.  size: n_items * array_size
 * @param [out] pos The positions of the new items.  size: n_items
 * @param [out] is_new Array where new elements are flagged as 1.
 *     size: n_items
 */
auto uniquelist_batched_unique_array_list_push_back(unique_array_list *list,
                                                  I array_size, I n_items,
                                                  const I *list_index,
                                                  const T *value, I *pos,
                                                  I *is_new) {
  const T* value_ = value;
  for (I i = 0; i < n_items; ++i) {
    // Use push_back_copy to avoid std::copy above.
    auto [_pos, _is_new] = list[list_index[i]].push_back(value_);
    pos[i] = _pos;
    is_new[i] = _is_new;
    value_ += array_size;
  }
}

/**
 * @brief Remove items whose values are smaller or equal to a given value
 *
 * @param [in] list Pointer to the array of lists.
 * @param [in] batch_size Number of lists.
 * @param [in] value A threshold value used in the test of deletion.
 */
auto uniquelist_batched_unique_array_list_erase_nonzero(unique_array_list *list,
                                                      I list_index, I flag_size,
                                                      I *flag) {
  return list[list_index].erase_nonzero(flag_size, flag);
}

/**
 * @brief Test whether given arrays are in the list or not
 *
 * @param [in] list Pointer to the array of unique_list.
 * @param [in] array_size Size of arrays in the lists.
 * @param [in] n_items Number of items to be added.
 * @param [in] list_index List indexes on which corresponding items
 *     are inserted.  size: n_items
 * @param [in] value A concatenation of values.  size: n_items * array_size
 * @param [out] result Array where found elements are flagged as 1.
 *     size: n_items
 */
auto uniquelist_batched_unique_array_list_isin(unique_array_list *list, I array_size,
                                             I n_items, const I *list_index,
                                             T *value, bool *result) {
  const T* value_ = value;
  for (I i = 0; i < n_items; ++i) {
    result[i] = list[list_index[i]].isin(value_);
    value_ += array_size;
  }
}

auto uniquelist_batched_unique_array_list_dump(unique_array_list *list, I list_index) {
  for (const auto &value : list[list_index]) {
    for (std::size_t i = 0; i < value.first; ++i) {
      std::cout << value.second[i] << " ";
    }
    std::cout << std::endl;
  }
}

} // extern C
