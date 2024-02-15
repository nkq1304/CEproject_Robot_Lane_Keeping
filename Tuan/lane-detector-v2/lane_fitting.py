import numpy as np
import cv2 as cv

from lane_line import LaneLine, Lane
from exceptions.lane import LeftLineNotFound, RightLineNotFound, LaneNotFound
from frame_debugger import FrameDebugger

class LaneFitting:
    def __init__(self, config: dict) -> None:
        self.nwindows = config.get('nwindows', 9)
        self.margin = config.get('margin', 100)
        self.minpix = config.get('minpix', 50)
        self.debug = config.get('debug', False)

    def fit(self, black_white_frame) -> Lane | None:
        img = black_white_frame.copy()
        self.leftx_base, self.rightx_base = self.get_starting_points(img)

        leftx_current = self.leftx_base
        rightx_current = self.rightx_base

        nwindows = self.nwindows
        margin = self.margin
        minpix = self.minpix

        window_height = img.shape[0] // nwindows
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if self.debug:
                cv.rectangle(img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv.rectangle(img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if self.validate_lane(leftx, lefty, rightx, righty) is False:
            self.visualize_lane(img, None)
            return

        left_line = LaneLine(np.polyfit(lefty, leftx, 2))
        right_line = LaneLine(np.polyfit(righty, rightx, 2))
        lane = Lane(left_line, right_line)

        self.visualize_lane(img, lane)

        return lane
    
    def validate_lane(self, leftx: np.ndarray, lefty: np.ndarray, rightx: np.ndarray, righty: np.ndarray) -> bool:
        left_lane_found = leftx.size != 0 and lefty.size != 0
        right_lane_found = rightx.size != 0 and righty.size != 0

        if left_lane_found and right_lane_found: return True

        if not left_lane_found and not right_lane_found:
            FrameDebugger.draw_error(LaneNotFound())
        elif not left_lane_found:
            FrameDebugger.draw_error(LeftLineNotFound())
        elif not right_lane_found:
            FrameDebugger.draw_error(RightLineNotFound())

        return False
    
    def visualize_lane(self, img, lane: Lane) -> None:
        if (self.debug is False):
            return
        
        if (lane is not None):
            ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
            left_fitx = lane.left_line.get_x(ploty)
            right_fitx = lane.right_line.get_x(ploty)

            for i in range(img.shape[0]):
                cv.circle(img, (int(left_fitx[i]), int(ploty[i])), 1, (255, 0, 0), 2)
                cv.circle(img, (int(right_fitx[i]), int(ploty[i])), 1, (0, 0, 255), 2)

        cv.imshow('lane_fitting', img)

    def get_starting_points(self, binary_warped) -> tuple:
        binary_warped = cv.cvtColor(binary_warped, cv.COLOR_BGR2GRAY)
        historgram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

        midpoint = historgram.shape[0] // 2
        leftx_base = np.argmax(historgram[:midpoint])
        rightx_base = np.argmax(historgram[midpoint:]) + midpoint

        return leftx_base, rightx_base

    