import numpy as np
import copy
import scipy.signal as ss
from numba import njit
from huri.learning.env.rack_v3.env import to_action

@njit
def _hs(num_tubetype, goalpattern, node):
    val = 0
    for i in range(1, num_tubetype + 1):
        val += np.sum((goalpattern != i) * (node == i))
    return val


def _to_action(place_id, pick_id, rack_size):
    return to_action(rack_size, pick_id, place_id)


mask_ucbc = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
mask_crcl = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
mask_ul = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]])
mask_ur = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]])
mask_bl = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])
mask_br = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])


# mask_ul = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
# mask_ur = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
# mask_bl = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
# mask_br = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]])


def get_fillable_movable(node: np.ndarray):
    cg_ucbc = ss.correlate2d(node, mask_ucbc)[1:-1, 1:-1]
    cg_crcl = ss.correlate2d(node, mask_crcl)[1:-1, 1:-1]
    cg_ul = ss.correlate2d(node, mask_ul)[1:-1, 1:-1]
    cg_ur = ss.correlate2d(node, mask_ur)[1:-1, 1:-1]
    cg_bl = ss.correlate2d(node, mask_bl)[1:-1, 1:-1]
    cg_br = ss.correlate2d(node, mask_br)[1:-1, 1:-1]
    cf = ((cg_ucbc == 0) + (cg_crcl == 0) + (cg_ul == 0) + (cg_ur == 0) + (cg_bl == 0) + (cg_br == 0)) * (
            node == 0)
    cg_ucbc[node == 0] = -1
    cg_crcl[node == 0] = -1
    cg_ul[node == 0] = -1
    cg_ur[node == 0] = -1
    cg_bl[node == 0] = -1
    cg_br[node == 0] = -1
    cg = (cg_ucbc == 0) + (cg_crcl == 0) + (cg_ul == 0) + (cg_ur == 0) + (cg_bl == 0) + (cg_br == 0)
    fillable_matrix = cf.astype(int)
    movable_matrix = cg.astype(int)
    return fillable_matrix, movable_matrix


def get_fillable_movable_condition_set(node: np.ndarray, conditional_set):
    condition_ucbc = conditional_set[..., 0]
    condition_crcl = conditional_set[..., 1]
    condition_ul = conditional_set[..., 2]
    condition_ur = conditional_set[..., 3]
    condition_bl = conditional_set[..., 4]
    condition_br = conditional_set[..., 5]
    cg_ucbc = ss.correlate2d(node, mask_ucbc)[1:-1, 1:-1]
    cg_crcl = ss.correlate2d(node, mask_crcl)[1:-1, 1:-1]
    cg_ul = ss.correlate2d(node, mask_ul)[1:-1, 1:-1]
    cg_ur = ss.correlate2d(node, mask_ur)[1:-1, 1:-1]
    cg_bl = ss.correlate2d(node, mask_bl)[1:-1, 1:-1]
    cg_br = ss.correlate2d(node, mask_br)[1:-1, 1:-1]
    cg_ucbc[condition_ucbc == 0] = 10
    cg_crcl[condition_crcl == 0] = 10
    cg_ul[condition_ul == 0] = 10
    cg_ur[condition_ur == 0] = 10
    cg_bl[condition_bl == 0] = 10
    cg_br[condition_br == 0] = 10
    cf = ((cg_ucbc == 0) + (cg_crcl == 0) + (cg_ul == 0) + (cg_ur == 0) + (cg_bl == 0) + (cg_br == 0)) * (
            node == 0)
    cg_ucbc[node == 0] = -1
    cg_crcl[node == 0] = -1
    cg_ul[node == 0] = -1
    cg_ur[node == 0] = -1
    cg_bl[node == 0] = -1
    cg_br[node == 0] = -1
    cg = (cg_ucbc == 0) + (cg_crcl == 0) + (cg_ul == 0) + (cg_ur == 0) + (cg_bl == 0) + (cg_br == 0)
    fillable_matrix = cf.astype(int)
    movable_matrix = cg.astype(int)
    return fillable_matrix, movable_matrix


class Node(object):

    def __init__(self, grid):

        """

        :param grid: np.array nrow*ncolumn

        author: weiwei
        date: 20190828, 20200104
        """

        self.grid = copy.deepcopy(grid)
        self._nrow, self._ncolumn = self.grid.shape
        self.ngrids = self._nrow * self._ncolumn
        self.parent = None
        self.gs = 0
        self.gridpadded = np.pad(self.grid, (1, 1), 'constant')

    @property
    def nrow(self):
        return self._nrow

    @property
    def ncolumn(self):
        return self._ncolumn

    def get3x3(self, i, j):
        """
        get the surrounding 3x3 mat at i,j

        :param i:
        :param j:
        :return:

        author: weiwei
        date: 20200425
        """

        i = i + 1
        j = j + 1
        return self.self.gridpadded[i - 1:i + 1, j - 1:j + 1]

    def __getitem__(self, x):
        return self.grid[x]

    def __eq__(self, anothernode):
        """
        determine if two nodes are the same
        :return:

        author: weiwei
        date: 20190828
        """

        return np.array_equal(self.grid, anothernode.grid)

    def __repr__(self):
        """
        overload the printed results

        :return:

        author: weiwei
        date: 20191003
        """

        outstring = "["
        for i in range(self._nrow):
            if i == 0:
                outstring += "["
            else:
                outstring += " ["
            for j in range(self._ncolumn):
                outstring = outstring + str(self.grid[i][j]) + ","
            outstring = outstring[:-1] + "]"
            outstring += ",\n"
        outstring = outstring[:-2] + "]]"

        return outstring


class TubePuzzle(object):

    def __init__(self, elearray, goalpattern=None):
        """

        :param nrow:
        :param ncolumn:
        :param elearray: nrow*ncolumn int array, tube id starts from 1, maximum 4

        author: weiwei
        date: 20191003
        """

        self._nrow = elearray.shape[0]
        self._ncolumn = elearray.shape[1]
        self.elearray = np.zeros((self._nrow, self._ncolumn), dtype="int")
        self.openlist = []
        self.closelist = []
        self._setValues(elearray)
        self.num_tubetype = 5
        if goalpattern is None:
            self.goalpattern = np.array([[1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
                                         [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
                                         [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
                                         [1, 1, 1, 0, 0, 0, 0, 2, 2, 2],
                                         [1, 1, 1, 0, 0, 0, 0, 2, 2, 2]])
        else:
            self.goalpattern = goalpattern

    def _setValues(self, elearray):
        """
        change the elements of the puzzle using elearray

        :param elearray: 2d array
        :return:

        author: weiwei
        date: 20190828, 20200104osaka
        """

        if elearray.shape != (self._nrow, self._ncolumn):
            print("Wrong number of elements in elelist!")
            raise Exception("Number of elements error!")
        self.elearray = elearray

    def isdone(self, node):
        """

        :return:

        author: weiwei
        date: 20190828
        """
        is_not_done = False
        for i in range(1, self.num_tubetype + 1):
            is_not_done = is_not_done or np.any((self.goalpattern != i) * (node.grid == i))
        if is_not_done:
            return False
        return True

    def fcost(self, node):
        hs = _hs(num_tubetype=self.num_tubetype, goalpattern=self.goalpattern, node=node.grid)
        gs = node.gs
        return hs + gs, hs, gs

    def getMovableFillablePair(self, node, weightarray, condition_set):
        """
        get a list of movable and fillable pairs

        :param node see Node
        :return: [[(i,j), (k,l)], ...]

        author: weiwei
        date: 20191003osaka, 20200104osaka
        """
        if condition_set is None:
            fillable, movable = get_fillable_movable(node.grid)
        else:
            fillable, movable = get_fillable_movable_condition_set(node.grid, condition_set)
        moveable_expanded_type_list = []
        fillable_expanded_type_list = []
        for tubetype in range(1, self.num_tubetype + 1):
            if tubetype not in node.grid:
                continue
            fillable_type = np.asarray(np.where((self.goalpattern * fillable) == tubetype)).T
            if len(fillable_type) == 0:
                continue
            # fillable_type = [fillable_type[np.random.choice(range(len(fillable_type)))]]
            fillable_type = [fillable_type[0]]
            # movable
            movable_type = np.asarray(np.where(((node.grid * movable) == tubetype))).T
            movable_expanded_type = np.repeat(movable_type, len(fillable_type), axis=0)
            fillable_expanded_type = np.tile(fillable_type, (len(movable_type), 1))
            if len(movable_expanded_type) > 0 and len(fillable_expanded_type) > 0:
                moveable_expanded_type_list.append(movable_expanded_type)
                fillable_expanded_type_list.append(fillable_expanded_type)
        if len(moveable_expanded_type_list) == 0 or len(fillable_expanded_type_list) == 0:
            return np.array([]), np.array([])
        movableeles = np.concatenate(moveable_expanded_type_list, axis=0)
        fillableeles = np.concatenate(fillable_expanded_type_list, axis=0)

        return movableeles, fillableeles

    def _reorderopenlist(self):
        self.openlist.sort(key=lambda x: (self.fcost(x)[0], self.fcost(x)[1]))

    def atarSearch(self,
                   condition_set=None,
                   infeasible_dict=None,
                   infeasible_set=None,
                   max_iter_cnt=1000):
        """

        build a graph considering the movable and fillable ids

        # CHEN HAO ADDED
        # Breadth First Search
        :param weightarray

        :return:

        author: weiwei
        date: 20191003
        """
        if infeasible_dict is None:
            infeasible_dict = {}
        if infeasible_set is None:
            infeasible_set = []
        # if weightarray is None:
        #     weightarray = np.zeros_like(self.elearray)

        startnode = Node(self.elearray)
        self.openlist = [startnode]
        iter_cnt = 0
        while True:
            iter_cnt += 1
            # print(iter_cnt)
            if iter_cnt >= max_iter_cnt:
                return False, []
            # if len(self.openlist)>=2:
            #     for eachnode in self.openlist:
            #         print(eachnode)
            #         print(eachnode.fcost())
            #     print("\n")
            self._reorderopenlist()
            # for opennode in self.openlist:
            #     print(opennode)
            #     print(self.fcost(opennode))
            #     print("\n")
            if len(self.openlist) > 0:
                self.closelist.append(self.openlist.pop(0))
            else:
                # print("open set cannot work!!!!")
                # print(self.closelist)
                tmpelearray = self.closelist[-1]
                path = [tmpelearray]
                parent = tmpelearray.parent
                while parent is not None:
                    path.append(parent)
                    parent = parent.parent
                # print("Path found!")
                # print(tmpelearray)
                # print(self.fcost(tmpelearray))
                # for eachnode in path:
                #     print(eachnode)
                return False, [_.grid for _ in path[::-1]]
            # movableids = self.getMovableIds(self.closelist[-1])
            # fillableids = self.getFillableIds(self.closelist[-1])
            # if len(movableids) == 0 or len(fillableids) == 0:
            #     print("No path found!")
            #     return []
            # for mid in movableids:
            #     for fid in fillableids:
            # todo consider weight array when get movable fillable pair
            movableeles, fillableeles = self.getMovableFillablePair(self.closelist[-1], None, condition_set)
            # print(movableeles)
            # print(fillableeles)
            if movableeles.shape[0] == 0:
                pass
                # print("No path found!")
                # return []
            for i in range(movableeles.shape[0]):
                mi, mj = movableeles[i]
                fi, fj = fillableeles[i]
                if _to_action(fillableeles[i], movableeles[i], self.elearray.shape) in infeasible_dict.get(
                        str(self.closelist[-1].grid), []) + infeasible_set:
                    continue
                # if weightarray[mi, mj] != 0 and weightarray[fi, fj] != 0:
                #     continue
                tmpelearray = copy.deepcopy(self.closelist[-1])
                tmpelearray.parent = self.closelist[-1]
                tmpelearray.gs = self.closelist[-1].gs + 1
                tmpelearray[fi][fj] = tmpelearray[mi][mj]
                tmpelearray[mi][mj] = 0
                #  check if path is found
                if self.isdone(tmpelearray) or (iter_cnt >= max_iter_cnt):
                    path = [tmpelearray]
                    parent = tmpelearray.parent
                    while parent is not None:
                        path.append(parent)
                        parent = parent.parent
                    # print("Path found!")
                    # print(tmpelearray)
                    # print(self.fcost(tmpelearray))
                    # for eachnode in path:
                    #     print(eachnode)
                    return iter_cnt < max_iter_cnt, [_.grid for _ in path[::-1]]
                # check if in openlist
                flaginopenlist = False
                for eachnode in self.openlist:
                    if eachnode == tmpelearray:
                        flaginopenlist = True
                        if self.fcost(eachnode)[0] <= self.fcost(tmpelearray)[0]:
                            pass
                            # no need to update position
                        else:
                            eachnode.parent = tmpelearray.parent
                            eachnode.gs = tmpelearray.gs
                            # self._reorderopenlist()
                        # continue
                        break
                if flaginopenlist:
                    continue
                else:
                    # not in openlist append and sort openlist
                    self.openlist.append(tmpelearray)


if __name__ == "__main__":
    # down x, right y
    elearray = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                         [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 2, 2],
                         [1, 0, 0, 0, 0, 0, 0, 2, 0, 2]])
    elearray = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 4, 4, 3, 0, 3, 0, 0, 4],
                         [2, 0, 0, 2, 1, 2, 0, 1, 0, 1],
                         [0, 1, 4, 1, 2, 3, 0, 3, 1, 3],
                         [0, 2, 0, 0, 2, 0, 2, 0, 0, 2]])

    # elearray = np.array([[0.0, 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0],
    #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3],
    #                      [0.0, 0.0, 3, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0],
    #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3, 0.0]]).astype(np.int)

    Node(elearray)

    tp = TubePuzzle(elearray)
    tp.goalpattern = np.array([[1, 1, 2, 2, 3, 3, 4, 4, 0, 3],
                               [1, 1, 2, 2, 3, 3, 4, 4, 0, 3],
                               [1, 1, 2, 2, 3, 3, 4, 4, 0, 3],
                               [1, 1, 2, 2, 3, 3, 4, 4, 0, 3],
                               [1, 1, 2, 2, 3, 3, 4, 4, 0, 3]])
    # tp.getMovableIds(Node(elearray))
    # print(Node(elearray).fcost())
    # print(tp.fcost(Node(elearray)))

    path = tp.atarSearch(max_iter_cnt=1000)
    print(path)
    # for node in path:////////////
    #     print(node)
