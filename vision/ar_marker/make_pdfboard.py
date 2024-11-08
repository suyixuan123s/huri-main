import numpy as np
from PIL import Image
from cv2 import aruco

_MM_TO_INCH = 0.0393701

def make_aruco_board(nrow,
                     ncolumn,
                     marker_dict=aruco.DICT_6X6_250,
                     start_id = 0,
                     marker_size=25,
                     savepath='./',
                     name='test',
                     frame_size = None,
                     paper_width=210,
                     paper_height=297,
                     dpi = 600):
    """
    create aruco board
    the paper is in portrait orientation, nrow means the number of markers in the vertical direction
    :param nrow:
    :param ncolumn:
    :param start_id: the starting id of the marker
    :param marker_dict:
    :param marker_size:
    :param savepath:
    :param name: the name of the saved pdf file
    :param frame_size: (width, height) the 1pt frame for easy cut, nothing is drawn by default
    :param paper_width: mm
    :param paper_height: mm
    :param dpi:
    :return:
    author: weiwei
    date: 20190420
    """
    aruco_dict = aruco.Dictionary_get(marker_dict)
    a4npxrow = int(paper_height * _MM_TO_INCH * dpi)
    a4npxcolumn = int(paper_width * _MM_TO_INCH * dpi)
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8')*255
    markersizepx = int(marker_size * _MM_TO_INCH * dpi)
    markerdist = int(markersizepx/4)
    if frame_size is not None:
        frame_size[0] = int(frame_size[0] * _MM_TO_INCH * dpi)
        frame_size[1] = int(frame_size[1] * _MM_TO_INCH * dpi)
        if a4npxcolumn<frame_size[0]+2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4npxrow<frame_size[1]+2:
            print("Frame height must be smaller than the #pt in each column.")
        framelft = int((a4npxcolumn - frame_size[0]) / 2 - 1)
        framergt = int(framelft + 1 + frame_size[0])
        frametop = int((a4npxrow - frame_size[1]) / 2 - 1)
        framedown = int(frametop + 1 + frame_size[1])
        bgimg[frametop:framedown+1, framelft:framelft+1]=0
        bgimg[frametop:framedown+1, framergt:framergt+1]=0
        bgimg[frametop:frametop+1, framelft:framergt+1]=0
        bgimg[framedown:framedown+1, framelft:framergt+1]=0
    markerareanpxrow = (nrow-1)*(markerdist)+nrow*markersizepx
    uppermargin = int((a4npxrow-markerareanpxrow)/2)
    markerareanpxcolumn = (ncolumn-1)*(markerdist)+ncolumn*markersizepx
    leftmargin = int((a4npxcolumn-markerareanpxcolumn)/2)
    if (uppermargin <= 10) or (leftmargin <= 10):
        print("Too many markers! Reduce nrow and ncolumn.")
        return
    for idnr in range(nrow):
        for idnc in range(ncolumn):
            startrow = uppermargin+idnr*(markersizepx+markerdist)
            endrow = startrow+markersizepx
            startcolumn = leftmargin+idnc*(markersizepx+markerdist)
            endcolumn = markersizepx+startcolumn
            i = start_id + idnr * ncolumn + idnc
            img = aruco.drawMarker(aruco_dict,i,markersizepx)
            bgimg[startrow:endrow, startcolumn:endcolumn] = img
    im = Image.fromarray(bgimg).convert("L")
    im.save(savepath+name+".pdf", "PDF", resolution=dpi)

def make_charuco_board(nrow,
                       ncolumn,
                       marker_dict=aruco.DICT_4X4_250,
                       square_size=25,
                       save_path='./',
                       name='test',
                       frame_size = None,
                       paper_width=210,
                       paper_height=297,
                       dpi = 600):
    """
    create charuco board
    the paper is in portrait orientation, nrow means the number of markers in the vertical direction
    :param nrow:
    :param ncolumn:
    :param marker_dict:
    :param save_path:
    :param name
    :param frame_size: (width, height) the 1pt frame for easy cut, nothing is drawn by default
    :param paper_width: mm
    :param paper_height: mm
    :param dpi:
    :return:
    author: weiwei
    date: 20190420
    """
    aruco_dict = aruco.Dictionary_get(marker_dict)
    # 1mm = _MM_TO_INCHinch
    a4npxrow = int(paper_height * _MM_TO_INCH * dpi)
    a4npxcolumn = int(paper_width * _MM_TO_INCH * dpi)
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8')*255
    if frame_size is not None:
        frame_size[0] = int(frame_size[0] * _MM_TO_INCH * dpi)
        frame_size[1] = int(frame_size[1] * _MM_TO_INCH * dpi)
        if a4npxcolumn<frame_size[0]+2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4npxrow<frame_size[1]+2:
            print("Frame height must be smaller than the #pt in each column.")
        framelft = int((a4npxcolumn - frame_size[0]) / 2 - 1)
        framergt = int(framelft + 1 + frame_size[0])
        frametop = int((a4npxrow - frame_size[1]) / 2 - 1)
        framedown = int(frametop + 1 + frame_size[1])
        bgimg[frametop:framedown+1, framelft:framelft+1]=0
        bgimg[frametop:framedown+1, framergt:framergt+1]=0
        bgimg[frametop:frametop+1, framelft:framergt+1]=0
        bgimg[framedown:framedown+1, framelft:framergt+1]=0
    squaresizepx = int(square_size * _MM_TO_INCH * dpi)
    squareareanpxrow = squaresizepx*nrow
    uppermargin = int((a4npxrow-squareareanpxrow)/2)
    squareareanpxcolumn = squaresizepx*ncolumn
    leftmargin = int((a4npxcolumn-squareareanpxcolumn)/2)
    if (uppermargin <= 10) or (leftmargin <= 10):
        print("Too many markers! Reduce nrow and ncolumn.")
        return
    board = aruco.CharucoBoard_create(ncolumn, nrow, square_size, .57 * square_size, aruco_dict)
    imboard = board.draw((squareareanpxcolumn, squareareanpxrow))
    print(imboard.shape)
    startrow = uppermargin
    endrow = uppermargin+squareareanpxrow
    startcolumn = leftmargin
    endcolumn = leftmargin+squareareanpxcolumn
    bgimg[startrow:endrow, startcolumn:endcolumn] = imboard
    im = Image.fromarray(bgimg).convert("L")
    im.save(save_path + name + ".pdf", "PDF", resolution=dpi)

def make_chess_board(nrow,
                     ncolumn,
                     square_size=25,
                     savepath='./',
                     name="test",
                     frame_size = None,
                     paper_width=210,
                     paper_height=297,
                     dpi = 600):
    """
    create checss board
    the paper is in portrait orientation, nrow means the number of markers in the vertical direction
    :param nrow:
    :param ncolumn:
    :param savepath:
    :param name
    :param frame_size: [width, height] the 1pt frame for easy cut, nothing is drawn by default
    :param paper_width: mm
    :param paper_height: mm
    :param dpi:
    :return:
    author: weiwei
    date: 20190420
    """
    a4npxrow = int(paper_height * _MM_TO_INCH * dpi)
    a4npxcolumn = int(paper_width * _MM_TO_INCH * dpi)
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8')*255
    if frame_size is not None:
        frame_size[0] = int(frame_size[0] * _MM_TO_INCH * dpi)
        frame_size[1] = int(frame_size[1] * _MM_TO_INCH * dpi)
        if a4npxcolumn<frame_size[0]+2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4npxrow<frame_size[1]+2:
            print("Frame height must be smaller than the #pt in each column.")
        framelft = int((a4npxcolumn - frame_size[0]) / 2 - 1)
        framergt = int(framelft + 1 + frame_size[0])
        frametop = int((a4npxrow - frame_size[1]) / 2 - 1)
        framedown = int(frametop + 1 + frame_size[1])
        bgimg[frametop:framedown+1, framelft:framelft+1]=0
        bgimg[frametop:framedown+1, framergt:framergt+1]=0
        bgimg[frametop:frametop+1, framelft:framergt+1]=0
        bgimg[framedown:framedown+1, framelft:framergt+1]=0
    squaresizepx = int(square_size * _MM_TO_INCH * dpi)
    squareareanpxrow = squaresizepx*nrow
    uppermargin = int((a4npxrow-squareareanpxrow)/2)
    squareareanpxcolumn = squaresizepx*ncolumn
    leftmargin = int((a4npxcolumn-squareareanpxcolumn)/2)
    if (uppermargin <= 10) or (leftmargin <= 10):
        print("Too many markers! Reduce nrow and ncolumn.")
        return
    for idnr in range(nrow):
        for idnc in range(ncolumn):
            startrow = uppermargin+idnr*squaresizepx
            endrow = startrow+squaresizepx
            startcolumn = leftmargin+idnc*squaresizepx
            endcolumn = squaresizepx+startcolumn
            if idnr%2 != 0 and idnc%2 != 0:
                bgimg[startrow:endrow, startcolumn:endcolumn] = 0
            if idnr%2 == 0 and idnc%2 == 0:
                bgimg[startrow:endrow, startcolumn:endcolumn] = 0
    im = Image.fromarray(bgimg).convert("L")
    im.save(savepath+name+".pdf", "PDF", resolution=dpi)
    worldpoints = np.zeros((nrow*ncolumn, 3), np.float32)
    worldpoints[:, :2] = np.mgrid[:nrow, :ncolumn].T.reshape(-1, 2) * square_size
    return worldpoints

def make_chess_and_charuco_board(nrow_chess=3,
                                 ncolumn_chess=5,
                                 square_size=25,
                                 nrowch_aruco=3,
                                 ncolumn_charuco=5,
                                 marker_dict=aruco.DICT_6X6_250,
                                 square_size_aruco=25,
                                 save_path='./',
                                 name='test',
                                 frame_size = None,
                                 paper_width=210,
                                 paper_height=297,
                                 dpi = 600):
    """
    create half-chess and half-charuco board
    the paper is in portrait orientation, nrow means the number of markers in the vertical direction
    :param nrow:
    :param ncolumn:
    :param square_size: mm
    :param marker_dict:
    :param save_path:
    :param name
    :param paper_width: mm
    :param paper_height: mm
    :param dpi:
    :param frame_size: (width, height) the 1pt frame for easy cut, nothing is drawn by default
    :return:
    author: weiwei
    date: 20190420
    """
    aruco_dict = aruco.Dictionary_get(marker_dict)
    a4npxrow = int(paper_height * _MM_TO_INCH * dpi)
    a4npxcolumn = int(paper_width * _MM_TO_INCH * dpi)
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8')*255
    if frame_size is not None:
        frame_size[0] = int(frame_size[0] * _MM_TO_INCH * dpi)
        frame_size[1] = int(frame_size[1] * _MM_TO_INCH * dpi)
        if a4npxcolumn<frame_size[0]+2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4npxrow<frame_size[1]+2:
            print("Frame height must be smaller than the #pt in each column.")
        framelft = int((a4npxcolumn - frame_size[0]) / 2 - 1)
        framergt = int(framelft + 1 + frame_size[0])
        frametop = int((a4npxrow - frame_size[1]) / 2 - 1)
        framedown = int(frametop + 1 + frame_size[1])
        bgimg[frametop:framedown+1, framelft:framelft+1]=0
        bgimg[frametop:framedown+1, framergt:framergt+1]=0
        bgimg[frametop:frametop+1, framelft:framergt+1]=0
        bgimg[framedown:framedown+1, framelft:framergt+1]=0
    # upper half, charuco
    squaresizepx = int(square_size_aruco * _MM_TO_INCH * dpi)
    squareareanpxrow = squaresizepx * nrow_chess
    uppermargin = int((a4npxrow/2-squareareanpxrow)/2)
    squareareanpxcolumn = squaresizepx * ncolumn_chess
    leftmargin = int((a4npxcolumn-squareareanpxcolumn)/2)
    if (uppermargin <= 10) or (leftmargin <= 10):
        print("Too many markers! Reduce nrow and ncolumn.")
        return
    board = aruco.CharucoBoard_create(ncolumn_chess, nrow_chess, square_size_aruco, .57 * square_size_aruco, aruco_dict)
    imboard = board.draw((squareareanpxcolumn, squareareanpxrow))
    print(imboard.shape)
    startrow = uppermargin
    endrow = uppermargin+squareareanpxrow
    startcolumn = leftmargin
    endcolumn = leftmargin+squareareanpxcolumn
    bgimg[startrow:endrow, startcolumn:endcolumn] = imboard
    # lower half, chess
    squaresizepx = int(square_size * _MM_TO_INCH * dpi)
    squareareanpxrow = squaresizepx * nrowch_aruco
    uppermargin = int((a4npxrow/2-squareareanpxrow)/2)
    squareareanpxcolumn = squaresizepx * ncolumn_charuco
    leftmargin = int((a4npxcolumn-squareareanpxcolumn)/2)
    if (uppermargin <= 10) or (leftmargin <= 10):
        print("Too many markers! Reduce nrow and ncolumn.")
        return
    for idnr in range(nrowch_aruco):
        for idnc in range(ncolumn_charuco):
            startrow = int(a4npxrow/2)+uppermargin+idnr*squaresizepx
            endrow = startrow+squaresizepx
            startcolumn = leftmargin+idnc*squaresizepx
            endcolumn = squaresizepx+startcolumn
            if idnr%2 != 0 and idnc%2 != 0:
                bgimg[startrow:endrow, startcolumn:endcolumn] = 0
            if idnr%2 == 0 and idnc%2 == 0:
                bgimg[startrow:endrow, startcolumn:endcolumn] = 0
    im = Image.fromarray(bgimg).convert("L")
    im.save(save_path + name + ".pdf", "PDF", resolution=dpi)

if __name__ == '__main__':
    # makechessandcharucoboard(4,6,32,5,7)
    # makecharucoboard(7,5, square_size=40)
    # makechessboard(7,5, square_size=40)
    # makearucoboard(2,2, marker_size=80)
    make_aruco_board(2, 2, marker_dict=aruco.DICT_4X4_250, start_id=1, marker_size=50, frame_size=[60, 60])
    # makechessboard(1, 1, square_size=35, frame_size = [100,150])