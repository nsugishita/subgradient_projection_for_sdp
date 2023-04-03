Python C++ extension to manage a list of indexes.

This is a C++ extension to manage a list of indexes.
In particular, this deals with deletion of indexes.
For example, let `a = [2, 5, 8, 9]` be the index array of some items in a list.

```
[ 0 1 2 3 4 5 6 7 8 9]
      ^     ^     ^ ^
```

Suppose we delete item at index 3, and 7.

```
[ 0 1 2 3 4 5 6 7 8 9]
      ^ X   ^   X ^ ^
```

After the deletion, the indexes will be shifted:

```
[ 0 1 2   3 4 5   6 7]
      ^     ^     ^ ^
```

Thus, we update `a = [2, 4, 6, 7]`.
This can be computed by `indexremove.remove(a, [3, 7])`.
