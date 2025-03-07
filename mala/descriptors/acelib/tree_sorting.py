"""Class and functions for building binary trees."""


class Node:
    """
    Class encapsulating a node in a binary tree.

    Parameters
    ----------
    i : any
        Value of this node.

    Attributes
    ----------
    value : any
        Value of this node.

    value_tuple : tuple
        Tuple of the value of this node and the values of its children.

    children : List
        List of the children of this node.

    leaves : List
        List of the leaves of this node.

    left : Node
        Left child of this node.

    right : Node
        Right child of this node.

    is_sorted : bool
        Whether the children of this node are sorted.

    parent : Node
        Parent of this node.

    tree_id : any
        Identifier of the tree this node belongs to.
    """

    def __init__(self, i):
        self.value = i
        self.value_tuple = None
        self.children = [None, None]
        self.leaves = []
        self.left = self.children[0]
        self.right = self.children[1]
        self.is_sorted = False
        self.parent = None
        self.tree_id = None

    def check_sorted(self, sort_depth=1):
        """
        Check if the children of this node are sorted.

        Parameters
        ----------
        sort_depth : int
            Depth to check sorting at.

        Returns
        -------
        is_sorted : bool
            Whether the children of this node are sorted.
        """
        self.is_sorted = False
        if sort_depth == 1:
            if self.left is None and self.right is None:
                self.is_sorted = True
            elif self.left is None and self.right is not None:
                self.is_sorted = False
            elif self.left is not None and self.right is None:
                self.is_sorted = False
            elif self.left is not None and self.right is not None:
                if self.left.value <= self.right.value:
                    self.is_sorted = True
                elif self.left.value > self.right.value:
                    self.is_sorted = False
            returnval = self.is_sorted
        elif sort_depth == 2:
            assert (
                self.left is not None and self.right is not None
            ), "Must have parent nodes to compare children"
            self.left.get_current_value_tuple()
            lv = self.left.value_tuple[1]
            self.right.get_current_value_tuple()
            rv = self.right.value_tuple[1]
            if self.left.value_tuple < self.right.value_tuple:
                if lv[0] <= lv[1] and rv[0] <= rv[1]:
                    self.is_sorted = True
                else:
                    if lv[0] > lv[1]:
                        self.left.flip_children()
                        self.left.get_current_value_tuple()
                        lv = self.left.value_tuple[1]
                        self.left.is_sorted = True
                    if rv[0] > rv[1]:
                        self.right.flip_children()
                        self.right.get_current_value_tuple()
                        rv = self.right.value_tuple[1]
                        self.right.is_sorted = True
                    self.is_sorted = True
            elif lv > rv:
                self.is_sorted = False
                if lv[0] > lv[1]:
                    self.left.flip_children()
                    self.left.get_current_value_tuple()
                    lv = self.left.value_tuple[1]
                    self.left.is_sorted = self.left.check_sorted()
                if rv[0] > rv[1]:
                    self.right.flip_children()
                    self.right.get_current_value_tuple()
                    rv = self.right.value_tuple[1]
                    self.right.is_sorted = self.right.check_sorted()
            returnval = (
                self.left.is_sorted and self.right.is_sorted and self.is_sorted
            )
        else:
            raise ValueError("Cannot sort deeper than depth 2 currently.")
        return returnval

    def return_children_vals(self, depth=1):
        """
        Return the values of the children of this node.

        Parameters
        ----------
        depth : int
            Depth up to which to return values.

        Returns
        -------
        vals : tuple
            Values of the children of this node.
        """
        if depth == 1:
            vals = (self.left.value, self.right.value)
        elif depth == 2:
            self.left.get_current_value_tuple()
            lv = self.left.value_tuple[1]
            self.right.get_current_value_tuple()
            rv = self.right.value_tuple[1]
            vals = lv + rv
        else:
            raise ValueError("Cannot sort deeper than depth 2 currently.")
        return vals

    def set_children(self, children, set_sorted=False, sort_depth=1):
        """
        Set the children of this node.

        Parameters
        ----------
        children : List or tuple
            Children to be set.

        set_sorted : bool
            If True, set the children in a sorted fashion.

        sort_depth : int
            Depth to sort children at, if set_sorted is True.
        """
        if isinstance(children, list):
            children = tuple(children)
        assert (
            len(children) == 2
        ), "list of children must be of length 2 for binary tree"
        if children[0] is not None and children[1] is not None:
            children[0].parent = self
            children[1].parent = self
            self.children[0] = children[0]
            self.children[1] = children[1]
            self.check_sorted(sort_depth)
            if set_sorted:
                if self.is_sorted:
                    if sort_depth == 1:
                        self.left = self.children[0]
                        self.right = self.children[1]
                    elif sort_depth == 2:
                        self.left = self.children[0]
                        self.right = self.children[1]
                        if self.left.is_sorted:
                            self.left.lft = self.left.children[0]
                            self.left.rght = self.left.children[1]
                        elif not self.left.is_sorted:
                            self.left.lft = self.left.children[1]
                            self.left.rght = self.left.children[0]
                            self.left.is_sorted = True
                        if self.right.is_sorted:
                            self.right.lft = self.right.children[0]
                            self.right.rght = self.right.children[1]
                        elif not self.right.is_sorted:
                            self.right.lft = self.right.children[1]
                            self.right.rght = self.right.children[0]
                            self.right.is_sorted = True

                elif not self.is_sorted:
                    if sort_depth == 1:
                        self.left = self.children[1]
                        self.right = self.children[0]
                    elif sort_depth == 2:
                        self.left = self.children[1]
                        self.right = self.children[0]
                        self.left.is_sorted = self.left.check_sorted()
                        self.right.is_sorted = self.right.check_sorted()
                        if self.left.is_sorted:
                            self.left.lft = self.left.children[0]
                            self.left.rght = self.left.children[1]
                        elif not self.left.is_sorted:
                            self.left.lft = self.left.children[1]
                            self.left.rght = self.left.children[0]
                            self.left.is_sorted = True
                        if self.right.is_sorted:
                            self.right.lft = self.right.children[0]
                            self.right.rght = self.right.children[1]
                        elif not self.right.is_sorted:
                            self.right.lft = self.right.children[1]
                            self.right.rght = self.right.children[0]
                            self.right.is_sorted = True
                    self.is_sorted = True

    def update_children(self, set_sorted=False, sort_depth=1):
        """
        Update the children of this node.

        Essentially calls set_children with children already attached to node.

        Parameters
        ----------
        set_sorted : bool
            If True, set the children in a sorted fashion.

        sort_depth : int
            Depth to sort children at, if set_sorted is True.
        """
        if self.left is not None and self.right is not None:
            children = [self.left, self.right]
            self.set_children(children, set_sorted, sort_depth=sort_depth)

    def get_current_value_tuple(self):
        """
        Return current value tuple of this node.

        Returns
        -------
        this_tup : tuple
            Tuple of the value of this node and the values of its children.
        """
        this_tup = (
            self.value,
            ((self.children[0].value), (self.children[1].value)),
        )
        self.value_tuple = this_tup
        return this_tup

    def flip_children(self):
        """Flip children (left/right) of node."""
        self.children.reverse()
        self.set_children(self.children)


def build_full_tree(l, L, L_R):
    """
    Build a full binary tree from the given lists.

    Works for rank 4, 5 and 6.

    Parameters
    ----------
    l : List
        list (multiset) of angular momentum indices l1,l2,...lN

    L : List
        list (multiset) of intermediate angular momentum indices l1,l2,...lN

    L_R : int
        Resultant angular momentum quantum number. This determines the equivariant
        character of the rank N descriptor after reduction. L_R=0 corresponds to
        a rotationally invariant feature, L_R=1 corresponds to a feature that
        transforms like a vector, L_R=2 a tensor, etc.

    Returns
    -------
    root : Node
        Root node of the constructed tree.
    """
    assert isinstance(l, list) or isinstance(
        l, tuple
    ), "convert l to list or tuple"
    assert isinstance(L, list) or isinstance(
        L, tuple
    ), "convert l to list or tuple"
    rank = len(l)

    def rank_2_binary(l_parent, l_left, l_right):
        """
        Build a rank 2 binary tree from the given lists.

        Parameters
        ----------
        l_parent : int
            Parent node value.

        l_left : int
            Value of the left child.

        l_right : int
            Value of the right child.

        Returns
        -------
        tree : tuple
            Tuple containing the root node and the leaves of the tree.
        """
        if isinstance(l_parent, int):
            root2 = Node(l_parent)
        else:
            root2 = l_parent
        root2.left = Node(l_left)
        root2.right = Node(l_right)
        root2.update_children(set_sorted=True)
        leaves = (root2.left.value, root2.right.value)
        return root2, leaves

    if rank == 4:
        root = Node(L_R)
        root.left, _ = rank_2_binary(L[0], l[0], l[1])
        root.right, _ = rank_2_binary(L[1], l[2], l[3])
        root.update_children(set_sorted=True, sort_depth=2)
        test_leaves = root.return_children_vals(depth=2)
        root.leaves = list(test_leaves)
    elif rank == 5:
        root = Node(L_R)
        root.left = Node(L[2])
        root.right = Node(l[4])
        root.left.left, _ = rank_2_binary(L[0], l[0], l[1])
        root.left.right, _ = rank_2_binary(L[1], l[2], l[3])
        root.left.update_children(set_sorted=True, sort_depth=2)
        test_leaves = root.left.return_children_vals(depth=2)
        root.leaves = list(test_leaves) + [l[4]]

    elif rank == 6:
        root = Node(L_R)
        root.left = Node(L[3])
        root.right, leaves_3 = rank_2_binary(L[2], l[4], l[5])
        root.right.update_children(set_sorted=True)
        root.update_children(set_sorted=True)
        root.left.left, _ = rank_2_binary(L[0], l[0], l[1])
        root.left.right, _ = rank_2_binary(L[1], l[2], l[3])
        root.left.update_children(set_sorted=True, sort_depth=2)
        test_leaves = root.left.return_children_vals(depth=2)
        root.leaves = list(test_leaves + leaves_3)

    else:
        raise ValueError(
            "rank %d not implemented yet for full tree construction" % rank
        )
    return root


def build_quick_tree(l):
    """
    Build a binary tree from the given list.

    Rank is assumed to be the length of the list. If the rank is odd, a
    remainder is returned.

    Parameters
    ----------
    l : List
        List to build the tree from.

    Returns
    -------
    tree : tuple
        Tuple containing the tree and the remainder which could not fit in the
        tree.
    """
    # quick construction of tree leaves
    rank = len(l)
    rngs = list(range(0, rank))
    rngs = iter(rngs)
    count = 0
    tup = []
    while count < int(rank / 2):
        c1 = next(rngs)
        c2 = next(rngs)
        tup.append((c1, c2))
        count += 1
    remainder = None
    if rank % 2 != 0:
        remainder = list(range(rank))[-1]
    return tuple(tup), remainder
