from mala.descriptors.label_sublib.young import *


class Node:
    def __init__(self, i):
        self.val = i
        self.val_tup = None
        self.children = [None, None]
        self.leaves = []
        self.lft = self.children[0]
        self.rght = self.children[1]
        self.is_sorted = False
        self.parent = None
        self.tree_id = None

    def set_parent(self, parent):
        self.parent = parent

    def set_leaves(self, leaves):
        self.leaves = leaves

    def check_sorted(self, sort_depth=1):
        self.is_sorted = False
        if sort_depth == 1:
            if self.lft is None and self.rght is None:
                self.is_sorted = True
            elif self.lft is None and self.rght is not None:
                self.is_sorted = False
            elif self.lft is not None and self.rght is None:
                self.is_sorted = False
            elif self.lft is not None and self.rght is not None:
                if self.lft.val <= self.rght.val:
                    self.is_sorted = True
                elif self.lft.val > self.rght.val:
                    self.is_sorted = False
            returnval = self.is_sorted
        elif sort_depth == 2:
            assert (
                self.lft is not None and self.rght is not None
            ), "Must have parent nodes to compare children"
            self.lft.current_val_tup()
            lv = self.lft.val_tup[1]
            self.rght.current_val_tup()
            rv = self.rght.val_tup[1]
            if self.lft.val_tup < self.rght.val_tup:
                if lv[0] <= lv[1] and rv[0] <= rv[1]:
                    self.is_sorted = True
                else:
                    if lv[0] > lv[1]:
                        self.lft.flip_children()
                        self.lft.current_val_tup()
                        lv = self.lft.val_tup[1]
                        self.lft.is_sorted = True
                    if rv[0] > rv[1]:
                        self.rght.flip_children()
                        self.rght.current_val_tup()
                        rv = self.rght.val_tup[1]
                        self.rght.is_sorted = True
                    self.is_sorted = True
            elif lv > rv:
                self.is_sorted = False
                if lv[0] > lv[1]:
                    self.lft.flip_children()
                    self.lft.current_val_tup()
                    lv = self.lft.val_tup[1]
                    self.lft.is_sorted = self.lft.check_sorted()
                if rv[0] > rv[1]:
                    self.rght.flip_children()
                    self.rght.current_val_tup()
                    rv = self.rght.val_tup[1]
                    self.rght.is_sorted = self.rght.check_sorted()
            returnval = (
                self.lft.is_sorted and self.rght.is_sorted and self.is_sorted
            )
        else:
            raise ValueError("Cannot sort deeper than depth 2 currently.")
        return returnval

    def return_children_vals(self, depth=1):
        if depth == 1:
            vals = (self.lft.val, self.rght.val)
        elif depth == 2:
            self.lft.current_val_tup()
            lv = self.lft.val_tup[1]
            self.rght.current_val_tup()
            rv = self.rght.val_tup[1]
            vals = lv + rv
        else:
            raise ValueError("Cannot sort deeper than depth 2 currently.")
        return vals

    def set_children(self, children, set_sorted=False, sort_depth=1):
        if isinstance(children, list):
            children = tuple(children)
        assert (
            len(children) == 2
        ), "list of children must be of length 2 for binary tree"
        if children[0] is not None and children[1] is not None:
            children[0].set_parent(self)
            children[1].set_parent(self)
            self.children[0] = children[0]
            self.children[1] = children[1]
            self.check_sorted(sort_depth)
            if set_sorted:
                if self.is_sorted:
                    if sort_depth == 1:
                        self.lft = self.children[0]
                        self.rght = self.children[1]
                    elif sort_depth == 2:
                        self.lft = self.children[0]
                        self.rght = self.children[1]
                        if self.lft.is_sorted:
                            self.lft.lft = self.lft.children[0]
                            self.lft.rght = self.lft.children[1]
                        elif not self.lft.is_sorted:
                            self.lft.lft = self.lft.children[1]
                            self.lft.rght = self.lft.children[0]
                            self.lft.is_sorted = True
                        if self.rght.is_sorted:
                            self.rght.lft = self.rght.children[0]
                            self.rght.rght = self.rght.children[1]
                        elif not self.rght.is_sorted:
                            self.rght.lft = self.rght.children[1]
                            self.rght.rght = self.rght.children[0]
                            self.rght.is_sorted = True

                elif not self.is_sorted:
                    if sort_depth == 1:
                        self.lft = self.children[1]
                        self.rght = self.children[0]
                    elif sort_depth == 2:
                        self.lft = self.children[1]
                        self.rght = self.children[0]
                        self.lft.is_sorted = self.lft.check_sorted()
                        self.rght.is_sorted = self.rght.check_sorted()
                        if self.lft.is_sorted:
                            self.lft.lft = self.lft.children[0]
                            self.lft.rght = self.lft.children[1]
                        elif not self.lft.is_sorted:
                            self.lft.lft = self.lft.children[1]
                            self.lft.rght = self.lft.children[0]
                            self.lft.is_sorted = True
                        if self.rght.is_sorted:
                            self.rght.lft = self.rght.children[0]
                            self.rght.rght = self.rght.children[1]
                        elif not self.rght.is_sorted:
                            self.rght.lft = self.rght.children[1]
                            self.rght.rght = self.rght.children[0]
                            self.rght.is_sorted = True
                    self.is_sorted = True

    def update_children(self, set_sorted=False, sort_depth=1):
        if self.lft is not None and self.rght is not None:
            children = [self.lft, self.rght]
            self.set_children(children, set_sorted, sort_depth=sort_depth)

    def current_val_tup(self):
        this_tup = (self.val, ((self.children[0].val), (self.children[1].val)))
        self.val_tup = this_tup
        return this_tup

    def flip_children(self):
        self.children.reverse()
        self.set_children(self.children)


def build_tree(l, L, L_R):
    assert isinstance(l, list) or isinstance(
        l, tuple
    ), "convert l to list or tuple"
    assert isinstance(L, list) or isinstance(
        L, tuple
    ), "convert l to list or tuple"
    rank = len(l)

    def rank_2_binary(l_parent, l_left, l_right):
        if isinstance(l_parent, int):
            root2 = Node(l_parent)
        else:
            root2 = l_parent
        root2.lft = Node(l_left)
        root2.rght = Node(l_right)
        root2.update_children(set_sorted=True)
        leaves = (root2.lft.val, root2.rght.val)
        return root2, leaves

    if rank == 4:
        root = Node(L_R)
        root.lft, _ = rank_2_binary(L[0], l[0], l[1])
        root.rght, _ = rank_2_binary(L[1], l[2], l[3])
        root.update_children(set_sorted=True, sort_depth=2)
        test_leaves = root.return_children_vals(depth=2)
        root.set_leaves(list(test_leaves))
    elif rank == 5:
        root = Node(L_R)
        root.lft = Node(L[2])
        root.rght = Node(l[4])
        root.lft.lft, _ = rank_2_binary(L[0], l[0], l[1])
        root.lft.rght, _ = rank_2_binary(L[1], l[2], l[3])
        root.lft.update_children(set_sorted=True, sort_depth=2)
        test_leaves = root.lft.return_children_vals(depth=2)
        # root.set_leaves(list(test_leaves)+(l[4],))
        root.set_leaves(list(test_leaves) + [l[4]])

    elif rank == 6:
        root = Node(L_R)
        root.lft = Node(L[3])
        root.rght, leaves_3 = rank_2_binary(L[2], l[4], l[5])
        root.rght.update_children(set_sorted=True)
        root.update_children(set_sorted=True)
        root.lft.lft, _ = rank_2_binary(L[0], l[0], l[1])
        root.lft.rght, _ = rank_2_binary(L[1], l[2], l[3])
        root.lft.update_children(set_sorted=True, sort_depth=2)
        test_leaves = root.lft.return_children_vals(depth=2)
        root.set_leaves(list(test_leaves + leaves_3))
    return root
