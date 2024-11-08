import numpy as np
import copy
import math
import cv2
import time
import scipy.signal as ss


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

    def _hs(self, node):
        """
        heuristics

        :return:

        author: weiwei
        date: 20200104
        """
        val = 0

        for i in range(1, self.num_tubetype + 1):
            val += np.sum((self.goalpattern != i) * (node.grid == i))
        return val

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
        hs = self._hs(node)
        gs = node.gs
        return hs + gs, hs, gs

    def getMovableFillablePair(self, node, weightarray):
        """
        get a list of movable and fillable pairs

        :param node see Node
        :return: [[(i,j), (k,l)], ...]

        author: weiwei
        date: 20191003osaka, 20200104osaka
        """

        # filtering
        # mask_ulbr = np.array([[1,0,0],[0,0,0],[0,0,1]])
        # mask_urbl = np.array([[0,0,1],[0,0,0],[1,0,0]])
        # mask_ulbr2 = np.array([[1,0,0],[1,0,0],[0,1,1]])
        # mask_urbl2 = np.array([[0,1,1],[1,0,0],[1,0,0]])
        # mask_ulbr2_flp = np.array([[1,1,0],[0,0,1],[0,0,1]])
        # mask_urbl2_flp = np.array([[0,0,1],[0,0,1],[1,1,0]])
        mask_ucbc = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
        mask_crcl = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
        mask_ul = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]])
        mask_ur = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]])
        mask_bl = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])
        mask_br = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
        # cg_ulbr = ss.correlate2d(node.grid, mask_ulbr)[1:-1,1:-1]
        # cg_urbl = ss.correlate2d(node.grid, mask_urbl)[1:-1,1:-1]
        # cg_ulbr2_flp = ss.correlate2d(node.grid, mask_ulbr2_flp)[1:-1,1:-1]
        # cg_urbl2_flp = ss.correlate2d(node.grid, mask_urbl2_flp)[1:-1,1:-1]
        # cg_ulbr2_flp = ss.correlate2d(node.grid, mask_ulbr2_flp)[1:-1,1:-1]
        # cg_urbl2_flp = ss.correlate2d(node.grid, mask_urbl2_flp)[1:-1,1:-1]
        cg_ucbc = ss.correlate2d(node.grid, mask_ucbc)[1:-1, 1:-1]
        cg_crcl = ss.correlate2d(node.grid, mask_crcl)[1:-1, 1:-1]
        cg_ul = ss.correlate2d(node.grid, mask_ul)[1:-1, 1:-1]
        cg_ur = ss.correlate2d(node.grid, mask_ur)[1:-1, 1:-1]
        cg_bl = ss.correlate2d(node.grid, mask_bl)[1:-1, 1:-1]
        cg_br = ss.correlate2d(node.grid, mask_br)[1:-1, 1:-1]
        ## fillable
        # cf = ((cg_ulbr==0)+(cg_urbl==0)+(cg_ulbr_flp==0)+(cg_urbl_flp==0)+(cg_ucbc==0)+(cg_crcl==0))*(node.grid==0)
        # cf = ((cg_ulbr==0)+(cg_urbl==0)+(cg_ucbc==0)+(cg_crcl==0))*(node.grid==0)
        # cf = ((cg_ucbc==0)+(cg_crcl==0))*(node.grid==0)
        cf = ((cg_ucbc == 0) + (cg_crcl == 0) + (cg_ul == 0) + (cg_ur == 0) + (cg_bl == 0) + (cg_br == 0)) * (
                node.grid == 0)
        # cf = ((cg_ul == 0) + (cg_ur == 0) + (cg_bl == 0) + (cg_br == 0)) * (node.grid == 0)
        # always fill the first element
        # # fillable 1
        # for i in range(np.asarray(np.where((self.goalpattern==1)*cf)).T.shape[0]):
        #     fillable_type1 = [np.asarray(np.where((self.goalpattern==1)*cf)).T[i]]
        #     if weightarray[fillable_type1[0][0], fillable_type1[0][1]] !=0:
        #         continue
        # # fillable 2
        # for i in range(np.asarray(np.where((self.goalpattern==2)*cf)).T.shape[0]):
        #     fillable_type2 = [np.asarray(np.where((self.goalpattern==2)*cf)).T[i]]
        #     if weightarray[fillable_type2[0][0], fillable_type2[0][1]] !=0:
        #         continue
        moveable_exoanded_type_list = []
        fillable_expanded_type_list = []
        for tubetype in range(1, self.num_tubetype + 1)[::-1]:
            if tubetype not in node.grid:
                continue
            fillable_type_all = np.asarray(np.where((self.goalpattern == tubetype) * cf)).T
            if len(fillable_type_all) == 0:
                continue
            # for fill_place in fillable_type_all:
            #     if weightarray[tuple(fill_place)]
            # fillable_type = fillable_type_all
            fillable_type = [fillable_type_all[0]]
            # cg_ucbc[node.grid == 0] = -1
            # cg_crcl[node.grid == 0] = -1
            cg_ul[node.grid == 0] = -1
            cg_ur[node.grid == 0] = -1
            cg_bl[node.grid == 0] = -1
            cg_br[node.grid == 0] = -1
            # cg = (cg_ulbr==0)+(cg_urbl==0)+(cg_ulbr_flp==0)+(cg_urbl_flp==0)+(cg_ucbc==0)+(cg_crcl==0)
            # cg = (cg_ulbr==0)+(cg_urbl==0)+(cg_ucbc==0)+(cg_crcl==0)
            # cg = (cg_ucbc==0)+(cg_crcl==0)
            # cg = (cg_ucbc == 0) + (cg_crcl == 0) + (cg_ul == 0) + (cg_ur == 0) + (cg_bl == 0) + (cg_br == 0)
            cg = (cg_ul == 0) + (cg_ur == 0) + (cg_bl == 0) + (cg_br == 0)
            # movable
            movable_type = np.asarray(np.where(cg * (node.grid == tubetype))).T
            tmp_fillable_type = fillable_type.copy()
            # np.random.shuffle(tmp_fillable_type)
            for fill_slot in tmp_fillable_type:
                for move_slot in movable_type:
                    if weightarray[tuple(move_slot)] != 0 and weightarray[tuple(fill_slot)] != 0:
                        continue
                    fillable_type = [fill_slot]
                    break
            movable_expanded_type = np.repeat(movable_type, len(fillable_type), axis=0)
            fillable_expanded_type = np.tile(fillable_type, (len(movable_type), 1))
            if len(movable_expanded_type) > 0 and len(fillable_expanded_type) > 0:
                moveable_exoanded_type_list.append(movable_expanded_type)
                fillable_expanded_type_list.append(fillable_expanded_type)
        if len(moveable_exoanded_type_list) == 0 or len(fillable_expanded_type_list) == 0:
            return np.array([]), np.array([])
        movableeles = np.concatenate(moveable_exoanded_type_list, axis=0)
        fillableeles = np.concatenate(fillable_expanded_type_list, axis=0)
        np.random.shuffle(movableeles)
        return movableeles, fillableeles

    def _reorderopenlist(self):
        self.openlist.sort(key=lambda x: (self.fcost(x)[0], self.fcost(x)[1]))

    def atarSearch(self, weightarray=None):
        """

        build a graph considering the movable and fillable ids

        # CHEN HAO ADDED
        # Breadth First Search
        :param weightarray

        :return:

        author: weiwei
        date: 20191003
        """

        if weightarray is None:
            weightarray = np.zeros_like(self.elearray)

        startnode = Node(self.elearray)
        self.openlist = [startnode]
        while True:
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
            self.closelist.append(self.openlist.pop(0))
            # movableids = self.getMovableIds(self.closelist[-1])
            # fillableids = self.getFillableIds(self.closelist[-1])
            # if len(movableids) == 0 or len(fillableids) == 0:
            #     print("No path found!")
            #     return []
            # for mid in movableids:
            #     for fid in fillableids:
            # todo consider weight array when get movable fillable pair
            movableeles, fillableeles = self.getMovableFillablePair(self.closelist[-1], weightarray)
            # print(movableeles)
            # print(fillableeles)
            if movableeles.shape[0] == 0:
                print("No path found!")
                # return []
            for i in range(movableeles.shape[0]):
                mi, mj = movableeles[i]
                fi, fj = fillableeles[i]
                if weightarray[mi, mj] != 0 and weightarray[fi, fj] != 0:
                    continue
                tmpelearray = copy.deepcopy(self.closelist[-1])
                tmpelearray.parent = self.closelist[-1]
                tmpelearray.gs = self.closelist[-1].gs + 1
                tmpelearray[fi][fj] = tmpelearray[mi][mj]
                tmpelearray[mi][mj] = 0
                #  check if path is found
                if self.isdone(tmpelearray):
                    path = [tmpelearray]
                    parent = tmpelearray.parent
                    while parent is not None:
                        path.append(parent)
                        parent = parent.parent
                    print("Path found!")
                    # print(tmpelearray)
                    # print(self.fcost(tmpelearray))
                    # for eachnode in path:
                    #     print(eachnode)
                    return path[::-1]
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


a_sol = [
    np.array([[0, 0, 1, 0, 0, 3, 0, 3, 0, 0],
              [0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
              [0, 0, 3, 0, 0, 1, 0, 1, 0, 0],
              [0, 0, 0, 3, 0, 0, 3, 0, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]),
    np.array([[0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
              [0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
              [0, 0, 3, 0, 0, 1, 0, 1, 0, 0],
              [0, 0, 0, 3, 0, 0, 3, 0, 1, 0],
              [0, 3, 0, 0, 0, 1, 0, 0, 0, 0]]),
    np.array([[0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
              [0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
              [0, 0, 3, 0, 0, 1, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 3, 0, 1, 0],
              [3, 3, 0, 0, 0, 1, 0, 0, 0, 0]]),
    np.array([[0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
              [0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
              [0, 3, 0, 0, 0, 0, 3, 0, 1, 0],
              [3, 3, 0, 0, 0, 1, 0, 0, 0, 0]]),
    np.array([[0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
              [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
              [3, 3, 0, 0, 0, 0, 3, 0, 1, 0],
              [3, 3, 0, 0, 0, 1, 0, 0, 0, 0]]),
    np.array([[0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 3, 0, 0, 0, 1, 0, 1, 0, 0],
              [3, 3, 0, 0, 0, 0, 3, 0, 1, 0],
              [3, 3, 0, 0, 0, 1, 0, 0, 0, 0]]),
    np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [3, 3, 0, 0, 0, 1, 0, 1, 0, 0],
              [3, 3, 0, 0, 0, 0, 3, 0, 1, 0],
              [3, 3, 0, 0, 0, 1, 0, 0, 0, 0]]),
    np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [3, 3, 0, 0, 0, 1, 0, 1, 0, 0],
              [3, 3, 0, 0, 0, 0, 3, 0, 1, 0],
              [3, 3, 0, 0, 0, 0, 0, 0, 0, 1]]),
    np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [3, 3, 0, 0, 0, 1, 0, 1, 0, 0],
              [3, 3, 0, 0, 0, 0, 3, 0, 1, 1],
              [3, 3, 0, 0, 0, 0, 0, 0, 0, 1]]),
    np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
              [3, 3, 0, 0, 0, 1, 0, 1, 0, 0],
              [3, 3, 0, 0, 0, 0, 0, 0, 1, 1],
              [3, 3, 0, 0, 0, 0, 0, 0, 0, 1]]),
    np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
              [3, 3, 0, 0, 0, 1, 0, 0, 0, 1],
              [3, 3, 0, 0, 0, 0, 0, 0, 1, 1],
              [3, 3, 0, 0, 0, 0, 0, 0, 0, 1]]),
    np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 3, 0, 0, 0, 0, 0, 0, 0, 1],
              [3, 3, 0, 0, 0, 0, 0, 0, 0, 1],
              [3, 3, 0, 0, 0, 0, 0, 0, 1, 1],
              [3, 3, 0, 0, 0, 0, 0, 0, 0, 1]]),
    np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 3, 0, 0, 0, 0, 0, 0, 0, 1],
              [3, 3, 0, 0, 0, 0, 0, 0, 0, 1],
              [3, 3, 0, 0, 0, 0, 0, 0, 0, 1],
              [3, 3, 0, 0, 0, 0, 0, 0, 0, 1]]),
    # np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [3, 3, 0, 0, 0, 1, 0, 1, 0, 0],
    #           [3, 3, 0, 0, 0, 0, 3, 0, 0, 0],
    #           [3, 3, 0, 0, 0, 1, 0, 0, 0, 1]]),
    # np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [3, 3, 0, 0, 0, 1, 0, 1, 0, 0],
    #           [3, 3, 0, 0, 0, 0, 3, 0, 0, 1],
    #           [3, 3, 0, 0, 0, 1, 0, 0, 0, 1]]),
    # np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [3, 3, 0, 0, 0, 1, 0, 1, 0, 1],
    #           [3, 3, 0, 0, 0, 0, 3, 0, 0, 1],
    #           [3, 3, 0, 0, 0, 0, 0, 0, 0, 1]]),
    # np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [3, 3, 0, 0, 0, 1, 0, 1, 0, 1],
    #           [3, 3, 0, 0, 0, 0, 0, 0, 0, 1],
    #           [3, 3, 0, 0, 0, 0, 0, 0, 0, 1]]),
    # np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 3, 0, 0, 0, 0, 0, 0, 0, 1],
    #           [3, 3, 0, 0, 0, 0, 0, 1, 0, 1],
    #           [3, 3, 0, 0, 0, 0, 0, 0, 0, 1],
    #           [3, 3, 0, 0, 0, 0, 0, 0, 0, 1]]),
    # np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #           [0, 3, 0, 0, 0, 0, 0, 0, 0, 1],
    #           [3, 3, 0, 0, 0, 0, 0, 0, 0, 1],
    #           [3, 3, 0, 0, 0, 0, 0, 0, 0, 1],
    #           [3, 3, 0, 0, 0, 0, 0, 0, 0, 1]]),
]

if __name__ == "__main__":
    # down x, right y
    elearray = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                         [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 2, 2],
                         [1, 0, 0, 0, 0, 0, 0, 2, 0, 2]])
    elearray = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [2, 2, 0, 2, 1, 0, 0, 0, 0, 0],
                         [1, 1, 0, 1, 2, 0, 0, 0, 0, 2],
                         [0, 2, 0, 0, 0, 0, 0, 0, 0, 2]])

    elearray = np.array([[0.0, 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3],
                         [0.0, 0.0, 3, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3, 0.0]]).astype(int)
    tp = TubePuzzle(elearray)
    tp.goalpattern = np.array([[1, 1, 2, 2, 3, 3, 4, 4, 0, 3],
                               [1, 1, 2, 2, 3, 3, 4, 4, 0, 3],
                               [1, 1, 2, 2, 3, 3, 4, 4, 0, 3],
                               [1, 1, 2, 2, 3, 3, 4, 4, 0, 3],
                               [1, 1, 2, 2, 3, 3, 4, 4, 0, 3]])
    # tp.getMovableIds(Node(elearray))
    # print(Node(elearray).fcost())
    # print(tp.fcost(Node(elearray)))
    path = tp.atarSearch()
    print(path)
    # for node in path:////////////
    #     print(node)
