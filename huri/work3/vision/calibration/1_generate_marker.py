import numpy as np
from PIL import Image
from cv2 import aruco

_MM_TO_INCH = 0.0393701


def make_aruco_board(nrow,
                     ncolumn,
                     marker_dict=aruco.DICT_6X6_250,
                     start_id=0,
                     marker_size=25,
                     savepath='./',
                     name='test',
                     frame_size=None,
                     paper_width=210,
                     paper_height=297,
                     marker_start_top_pos=None,
                     marker_start_lft_pos=None,
                     bg_img=None,
                     dpi=600):
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
    if bg_img is None:
        bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8') * 255
    else:
        bgimg = bg_img
    markersizepx = int(marker_size * _MM_TO_INCH * dpi)
    markerdist = int(markersizepx / 4)

    markerareanpxrow = (nrow - 1) * (markerdist) + nrow * markersizepx
    if marker_start_top_pos is None:
        uppermargin = int((a4npxrow - markerareanpxrow) / 2)
    else:
        uppermargin = int(marker_start_top_pos * _MM_TO_INCH * dpi)
    markerareanpxcolumn = (ncolumn - 1) * (markerdist) + ncolumn * markersizepx
    if marker_start_lft_pos is None:
        leftmargin = int((a4npxcolumn - markerareanpxcolumn) / 2)
    else:
        leftmargin = int(marker_start_lft_pos * _MM_TO_INCH * dpi)
    # if (uppermargin <= 5) or (leftmargin <= 5):
    #     print("Too many markers! Reduce nrow and ncolumn.")
    #     return

    if frame_size is not None:
        frame_size[0] = int(frame_size[0] * _MM_TO_INCH * dpi)
        frame_size[1] = int(frame_size[1] * _MM_TO_INCH * dpi)
        if a4npxcolumn < frame_size[0] + 2:
            print("Frame width must be smaller than the #pt in each row.")
        if a4npxrow < frame_size[1] + 2:
            print("Frame height must be smaller than the #pt in each column.")
        framelft = int((a4npxcolumn - frame_size[0]) / 2 - 1)
        framelft = max(leftmargin - markerdist, 0)
        framergt = int(framelft + 1 + frame_size[0])
        frametop = int((a4npxrow - frame_size[1]) / 2 - 1)
        frametop = max(uppermargin - markerdist, 0)
        framedown = int(frametop + 1 + frame_size[1])
        bgimg[frametop:framedown + 1, framelft:framelft + 1] = 0
        bgimg[frametop:framedown + 1, framergt:framergt + 1] = 0
        bgimg[frametop:frametop + 1, framelft:framergt + 1] = 0
        bgimg[framedown:framedown + 1, framelft:framergt + 1] = 0

    for idnr in range(nrow):
        for idnc in range(ncolumn):
            startrow = uppermargin + idnr * (markersizepx + markerdist)
            endrow = startrow + markersizepx
            startcolumn = leftmargin + idnc * (markersizepx + markerdist)
            endcolumn = markersizepx + startcolumn
            i = start_id + idnr * ncolumn + idnc
            img = aruco.drawMarker(aruco_dict, i, markersizepx)
            bgimg[startrow:endrow, startcolumn:endcolumn] = img
    im = Image.fromarray(bgimg).convert("L")
    im.save(savepath + name + ".pdf", "PDF", resolution=dpi)
    return bgimg


if __name__ == '__main__':
    lft_shift = 10
    top_shift = 10
    marker_size = 20
    marker_margin = 5
    marker_n_rows = 4
    im = make_aruco_board(nrow=marker_n_rows, ncolumn=1, marker_dict=aruco.DICT_4X4_250, start_id=1,
                          marker_size=marker_size,
                          marker_start_top_pos=marker_margin + top_shift,
                          marker_start_lft_pos=marker_margin + lft_shift,
                          frame_size=[marker_size + 2 * marker_margin,
                                      marker_size * marker_n_rows + (marker_n_rows + 1) * marker_margin])
    # im2 = make_aruco_board(nrow=2, ncolumn=1, marker_dict=aruco.DICT_4X4_250, start_id=3, marker_size=marker_size,
    #                        marker_start_top_pos=15,
    #                        marker_start_lft_pos=210 - marker_size - marker_margin, bg_img=im)
    print(im)
