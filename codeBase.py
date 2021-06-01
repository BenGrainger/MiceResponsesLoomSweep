import numpy as np
# https://numpy.org/doc/stable/user/numpy-for-matlab-users.html ~ useful info for you
import scipy
from scipy import signal
import scipy.io as sio
import os
import cv2
import math
import pickle
import pandas as pd


def open_csv(csv_loc):
    return pd.read_csv(look_up, index_col=False)


def get_matfile_locations(mat_file_loc):
    mat_file_locs = []
    for item in os.listdir(mat_file_loc):
        if '.mat' in item:
            mat_file_locs.append(mat_file_loc + '/' + item)
    return mat_file_locs


def get_frame_numbers_for_stim_presentations(csv):
    time_list = []
    calibration_files = csv['CalibrationFileName']

    for item in csv.columns:
        if 'FRAME' in item:
            time_list.append(item)

    dict_escapeFRAMES = {}

    for i in range(len(csv.index)):
        escape_frames = list(csv.loc[i, time_list])
        vid_name = look_up_df['VideoFileName'][i]
        dict_escapeFRAMES[vid_name] = escape_frames
    return dict_escapeFRAMES


def loadMATfile(MATfile, corners=[]):
    """
    loads the MAT file containing the checkers corners coordinates
    returns the coordinates in a calable form for CV2
    """
    loadedMATfile = sio.loadmat(MATfile)
    CORNERS = loadedMATfile['CALIBDATA'][0][0][3]

    for corner in CORNERS:
        xy = list(corner)
        xy.append(1)
        corners.append(xy)

    corners = np.array(corners, dtype='float32')
    return corners


def getManuallyAdjustedCorners(numberCorners, boardwidth=7, boardheight=8, xmax=500, ymax=580):
    """
    returns a set of cooridnates that correspond with the real corners
    however unsquewed by perspective or barrel distortion
    """
    widthPerc = 1 / (boardwidth - 1)
    heightPerc = 1 / (boardheight - 1)

    transform_cornersX = []
    transform_cornersY = []
    transform_corners = []

    for i in range(numberCorners):
        if i == 0:
            width = 0
            height = 0
        else:
            width = i % boardwidth
            height = math.floor(i / boardwidth)
        transform_cornersX.append(width * widthPerc * xmax)
        transform_cornersY.append(height * heightPerc * ymax)

    transform_cornersY.reverse()
    for x, y in zip(transform_cornersX, transform_cornersY):
        transform_corners.append([x, y])
    transform_corners = np.float32(transform_corners)
    return transform_corners


def PerspectiveDistortionMatrix(calibrationImage, corners, ManualCorners, warpingDims=(650, 800)):
    """
    generates linear matrix from the adjusted coordinates
    """
    img = cv2.imread(calibrationImage)
    rows, cols, ch = img.shape
    pts1 = np.array([[list(corners[49][:2])], [list(corners[-1][:2])], [list(corners[0][:2])], [list(corners[6][:2])]],
                    dtype='float32')
    pst2 = np.array([[list(ManualCorners[49] + 100)], [list(ManualCorners[-1] + 100)], [list(ManualCorners[0] + 100)],
                     [list(ManualCorners[6] + 100)]], dtype='float32')
    M = cv2.getPerspectiveTransform(pts1, pst2)

    return M


def save_numpy_array_singleArray(path):
    index = []
    numpy_file = []
    directory_list = os.listdir(path)
    for directory in directory_list:
        file_list = os.listdir(path + directory)
        for file in file_list:
            if 'escape00' in file:
                # os.remove(path + directory + '/' + file)

                index.append(file)
                save_file_name = path + directory + '/' + file
                numpy_file.append(list(np.load(save_file_name)))
    file_to_save = np.array(numpy_file)
    np.save('/home/beng/Desktop/allEscapes.npy', file_to_save, allow_pickle=True)


def FindJumps(coor, jumpThreshold):
    """
    input: coordinate, either x or y's
    find large jumps in tracking
    """

    rolledCoor = np.roll(coor, 1)
    frameDifferences = abs(rolledCoor[1:] - coor[1:])
    frameJumps = list(np.where(frameDifferences > jumpThreshold)[0])
    return frameJumps


def medianFilteringInterpolation(coors):
    """
    first applies median filtering
    then applies average interpolation over the larger gaps
    good explanation of this function: https://stackoverflow.com/questions/6518811
    good explanation of lamda function: https://realpython.com/python-lambda
    """

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    scipy.signal.medfilt(coors, 3)
    nans, x = nan_helper(coors)
    if False in nans:
        coors[nans] = np.interp(x(nans), x(~nans), coors[~nans])


def modified_ProcessRawData(rawData, bodyPartList, liklihoodIndex, problematicEscapes=[], LiklihoodThreshold=0.90,
                            houseBoundary=30, jumpThreshold=8, crossFrameThresh=70):
    """
    input: rawData, list of body part indexes for array slicing, liklihood index, various thresholds
    output: processed responses
    (1) removes low-liklihood frames (2) removes large frame jumps (3) applies median filtering (4) apply the savgol gaussian filter (5) removews house frames
    """

    processedEscapeList = []
    for responseIndex, response in enumerate(rawData):
        # iterate over each escape
        escapeColumnsForStacking = []
        for bodyPart, likely in zip(bodyPartList, liklihoodIndex):
            # iterate over each body part
            bodyPartCoors = response[bodyPart]
            X = np.copy(bodyPartCoors[0])
            Y = np.copy(bodyPartCoors[1])
            likelyVals = np.copy(response[liklihoodIndex[likely]])

            # find where tracking goes awry
            lowLiklihood = list(np.where(likelyVals < LiklihoodThreshold)[0])

            framesToRemove = list(set(lowLiklihood + FindJumps(X, jumpThreshold) + FindJumps(Y, jumpThreshold)))

            # removing those frames from cordinates
            for i in framesToRemove:
                X[i] = np.nan
                Y[i] = np.nan

            medianFilteringInterpolation(Y)
            medianFilteringInterpolation(X)

            escapeColumnsForStacking.append(X)
            escapeColumnsForStacking.append(Y)
            escapeColumnsForStacking.append(likelyVals)

            ProcessedEscape = list(np.column_stack(escapeColumnsForStacking).T)
        processedEscapeList.append(ProcessedEscape)
    processedEscape = np.array(processedEscapeList)
    return processedEscape


def AverageCoordinate(coor, bodyPartList, numberOfFRames, escapeNumber, escape):
    """
    input: coor 0 = X, 1 = Y, number of rows of the array corresponding to frames
    return the average coordinates across all body parts
    """

    coorAdd = np.zeros(numberOfFRames)
    for bodyPart in bodyPartList:
        bodyPartCoors = escape[bodyPart]
        coorAdd += np.copy(bodyPartCoors[coor])
    coorMean = coorAdd / len(bodyPartList)
    return coorMean


def createMeanTrack(ProcessedEscape, bodyPartList, averageEscapeList=[], numberOfFRames=frames):
    """
    Creates an array of the average tracks across all body parts for every escape
    """

    averageEscapeList = []
    for responseIndex, response in enumerate(ProcessedEscape):
        escapeAveragesStacking = []
        XMean = AverageCoordinate(0, bodyPartList, numberOfFRames, responseIndex, response)
        YMean = AverageCoordinate(1, bodyPartList, numberOfFRames, responseIndex, response)
        escapeAveragesStacking.append(XMean)
        escapeAveragesStacking.append(YMean)
        escapeAverage = list(np.column_stack(escapeAveragesStacking).T)
        averageEscapeList.append(escapeAverage)
    return np.array(averageEscapeList)


def find_angle_from_3_points(A, B, C, x=0, y=1):
    """
    input points/2D coordinates A B and C
    finding the angle of points A --> B --> C. 2 vectors: BA and BC
    outputs angle in degrees
    explanation here: https://math.stackexchange.com/questions/361412
    """

    AB = B - A
    BC = C - B
    vecAB = np.sqrt((A.T[x] - B.T[x]) ** 2 + (A.T[y] - B.T[y]) ** 2)
    vecBC = np.sqrt((B.T[x] - C.T[x]) ** 2 + (B.T[y] - C.T[y]) ** 2)
    dots = []
    for ab, bc in zip(AB, BC):
        dots.append(np.dot(ab, bc))
    dot = np.array(dots)
    denominator = (vecAB * vecBC)
    cosine_angle = dot / denominator

    # inverse cosine has a domain of -1 to 1
    # however cosine (the input) has perodicity
    # therefore if an input is outside either I take a period/ interval away or add one in

    angles = []
    for InvCosInput in cosine_angle:
        if (InvCosInput >= -1) & (InvCosInput <= 1):
            angle = np.arccos(InvCosInput)
            angle = np.degrees(angle)
            angles.append(float(angle))
        elif InvCosInput > 1:
            angle = np.arccos(InvCosInput - 2)
            angle = np.degrees(angle)
            angles.append(float(angle))
        elif InvCosInput < -1:
            angle = np.arccos(InvCosInput + 2)
            angle = np.degrees(angle)
            angles.append(float(angle))

    return np.array(angles)


def find_average_speed(mean, numberCoordinates):
    """
    input: mean cordinates slice
    outputs: value of average speed
    """

    nminusnplus1 = mean.T - np.roll(mean.T, 1)
    xdiff = nminusnplus1[0][1:]
    ydiff = nminusnplus1[1][1:]
    framebyframeSpeed = np.sqrt(np.add(xdiff ** 2, ydiff ** 2))
    return sum(framebyframeSpeed) / numberCoordinates


def calc_dist(back, front):
    """
    helper func for get_spine_length
    Calculates euclidean distance, between pairs of body parts
    """

    x = (back[0] - front[0]) ** 2
    y = (back[1] - front[1]) ** 2
    return np.sqrt(x + y)


def get_spine_length(response, frame, spineList):
    """
    on a frame by frame basis cycles through the vertebrae of the mice calculating the distances and summing them
    returns the length of the animal
    """

    length = 0

    # loop through the spine segments
    for vert1, vert2 in zip(spineList[:-1], spineList[1:]):
        spineLength = calc_dist(response[vert1].T[frame], response[vert2].T[frame])
        length += spineLength
    return length


def calc_head_deflections(response, frame, bodyPartList, def_angle_range=3):
    """
    calculate the angle between the 3 points A(nose), B(neck), C(upper spine) for the rnage of frames controlled by def_angle_range - harcoded as 3
    this defines head delections
    """

    nose = bodyPartList[0]
    neck = bodyPartList[3]
    spine1 = bodyPartList[4]
    delection_angles = find_angle_from_3_points(response[nose].T[frame:frame + def_angle_range],
                                                response[neck].T[frame:frame + def_angle_range],
                                                response[spine1].T[frame:frame + def_angle_range])

    # average over array of the absolute values of the head deflections
    Head_deflection = np.mean(abs(delection_angles[:-1] - delection_angles[1:]))
    return Head_deflection


def calc_body_rotations(mean, front, frame):
    """
    using the angle between 3 points function.
    a = front of the animal
    b = middle of the animal
    c = front of the animal but from the next frame
    """

    a = front.T[frame:frame + 5][1:] - mean.T[frame:frame + 5][1:]
    b = mean.T[frame:frame + 5][1:] - mean.T[frame:frame + 5][1:]
    c = np.roll(front.T[frame:frame + 5], 1, axis=0)[1:] - np.roll(mean.T[frame:frame + 5], 1, axis=0)[1:]
    Rotation = np.mean(180 - find_angle_from_3_points(a, b, c))
    return Rotation


def create_Event_array(event_occurance, array_length=frames):
    """
    input: when did event occur
    output: a zero vector (length representing the frames) with a 1 where the event occured
    """
    zero_array = np.zeros(array_length)
    if event_occurance == 0:
        # no event occured, return zero vector
        return zero_array

    else:
        zero_array[event_occurance] = 1
        return zero_array


def find_freeze(speed_data, startFrame=60, endFrame=270, speedThreshold=.5, duration=30):
    """
    loop through the data in 30 frame periods, if there is a sustained period of low activity, look for the lowest speed in that period
    input: index, speed data
    outputs: the frame where the freeze occurs
    """
    for i in range(startFrame, endFrame):
        slices = speed_data[i: i + duration]
        average = np.mean(slices)
        if average < speedThreshold:
            when = np.where(slices == np.min(slices))[0][0] + i
            return when

    else:
        # no freeze occured, this is passed into create event array so that it will be recorded as a non event
        return 0


def find_peak_speed(mean_coordinates):
    """
    same as find average speed, except finds max value rather than averaging
    """
    nminusnplus1 = mean_coordinates.T - np.roll(mean_coordinates.T, 1)
    xdiff = nminusnplus1[0][1:]
    ydiff = nminusnplus1[1][1:]
    framebyframeSpeed = np.sqrt(np.add(xdiff ** 2, ydiff ** 2))
    return max(framebyframeSpeed)


def return_response_type(mean, speed_data, stim_onset=60,
                         house_locs=[(200, 0), (225, 0), (250, 0), (275, 0), (300, 0), (325, 0), (350, 0), (375, 0),
                                     (400, 0)], speedThreshold=0.33):
    """
    outputs a matrix where: row 1 = escape, row 2 = freeze, row 3 = late return, zero matrix is a 'non-responder'
    """

    dist_from_house = []

    # emptry vector to create matrix rows later
    empty = np.zeros(frames)

    for frame in range(60, 290):
        distShelter = []
        # loop through house locations and check minimum
        for loc in house_locs:
            distShelter.append(calc_dist(mean.T[frame], loc))
        distShelter = min(distShelter)
        dist_from_house.append(distShelter)

    dist_from_house = np.array(dist_from_house)

    ## yardstick for comparing increases in speed

    house_boundary = 100

    # find if there is a house cross
    house_cross_new = np.where(dist_from_house < house_boundary)

    # if there is a house cross...
    if len(house_cross_new[0]) > 0:
        # yes, then check if there was a high speed event before the house cross
        house_cross = house_cross_new[0][0] + 60
        potentialEscapePeriod = np.copy(mean.T[house_cross - 20:house_cross])
        peakSpeed = find_peak_speed(potentialEscapePeriod)

        if peakSpeed > 5:

            if house_cross < 120:
                # must be within the first second after to be considered escape
                Escape = create_Event_array(house_cross)
                return np.vstack([Escape, empty, empty])

            # if no escape, check if there was a freeze before house return
            freezeEvent = find_freeze(speed_data[:house_cross], startFrame=60, endFrame=house_cross - 30)
            Freeze = create_Event_array(freezeEvent)

            # else this must have been a late return
            if freezeEvent != 0:
                return np.vstack([empty, Freeze, empty])

            else:
                LateReturn = create_Event_array(house_cross)
                return np.vstack([empty, empty, LateReturn])

        # no escape, them check if there was a freeze before the return
        else:
            baseline = np.mean(speed_data[20:50])
            threshold = baseline * speedThreshold
            freezeEvent = find_freeze(speed_data)
            Freeze = create_Event_array(freezeEvent)
            return np.vstack([empty, Freeze, empty])

    # no house cross check if there was an freeze event
    else:
        freezeEvent = find_freeze(speed_data)
        Freeze = create_Event_array(freezeEvent)
        return np.vstack([empty, Freeze, empty])


def create_event_occuranceVec(data, mxmn, thresh_above_baseline):
    """
    input: data type for a single response, maximum of minum, can be 'max' or 'min'
    threshold for the event
    outputs a vector of occurance of an event
    """

    zeros = np.zeros(frames)

    # slice of data of post stimulus behvaiour
    post_stimulus = data[60:120]

    # slice of data of pre stimulus behvaiour - for basline
    baseline = np.max(data[30:60])

    # create threshold as a proportion of baseline
    thresh = thresh_above_baseline * baseline

    # assigninig variable just due to reference issue in the block below - ignore this is hacky
    occurence = []

    # mxmn value determines whether the data should be more or less than the threshold
    if mxmn == 'max':
        occurence = np.where(post_stimulus > thresh)[0]

    # check if there was an occurance
    if len(occurence) > 0:

        # add on the initial frames
        occurence += 60

        if mxmn == 'max':
            zeros[occurence[0]] = 1
            return zeros

    elif mxmn == 'min':
        # function re-used for non speed releated data to look for minimum point
        occurance = find_freeze(data, 60, 115, speedThreshold=thresh, duration=10)
        event = create_Event_array(occurance)
        return event
    else:
        return zeros


def TransformDLC_output(csv_loc, mat_file_loc, path):
    """

    """
    csv = open_csv(csv_loc)

    mat_file_locs = get_matfile_locations(mat_file_loc)

    dict_escapeFRAMES = get_frame_numbers_for_stim_presentations(csv)

    directory_list = os.listdir(path)

    df = pd.DataFrame(columns=['animal+session', 'stimulus'])

    for directory in directory_list:
        ###
        # list directories from the directory list
        ###
        file_list = os.listdir(path + directory)
        for file in file_list:
            ##
            # list files inside the directory
            ##
            if '.h5' in file:
                ##
                # find the deeplabcut generated files by looking for H5 files
                ##
                dlc_path = path + directory + '/' + file
                ##
                # remove the name of the deeplabcut file to get the session number
                ##
                session_name = file.replace('DLC_resnet50_EscapeDec3shuffle1_1030000.h5', '')

                for i, (calibration_file, session_key) in enumerate(zip(calibration_files, dict_escapeFRAMES.keys())):
                    ##
                    # iterate through the calibration file names at the same level in the CSV as the excape frames
                    ##
                    if session_key == session_name:
                        ##
                        # find the matching key to the session name to find the correct calibration file and escape frames
                        ##
                        loaded = pd.read_hdf(dlc_path)
                        obstructive_title = loaded.keys()[0][0]
                        extracted_tracks = loaded[obstructive_title]
                        DLC_array = extracted_tracks.to_numpy()
                        ##
                        # load the H5 file and convert it to a rectangular array to be sliced later
                        ##

                        for escape_number, escapeFrame in enumerate(dict_escapeFRAMES[session_key]):
                            ##
                            # iterate through the escape frames
                            ##
                            if math.isnan(escapeFrame) == True:
                                ##
                                # if the escape frame is empty - i.e. no stimulus - skip
                                ##
                                continue

                            else:
                                ##
                                # if escape not empty load the associated mat file for calibration
                                ##
                                stim_number = 'escape00' + str(escape_number)

                                save_file_name = path + directory + '/' + session_name + stim_number + '.npy'

                                img = mat_file_loc + '/' + calibration_file
                                file_name = mat_file_loc + '/' + calibration_file + '.mat'
                                ##
                                # loadMATfile extracts the corners from the .mat file and converts them into 3D coordinates
                                ##
                                CORNER = loadMATfile(file_name)

                                ##
                                # slice the DLC numpy array from frame 1 of the stimulus to 3 seconds later, i.e. 240 frames
                                ##
                                frame = int(escapeFrame)
                                escape_array = DLC_array[frame:frame + 240]
                                escape_array = escape_array.T

                                ##
                                # count number of corners
                                ##
                                number_of_corners = len(CORNER)

                                if number_of_corners == 56:
                                    ##
                                    # if 56 corners counted, this value means the checkerboard was 7 wide and 8 high. this is important
                                    # allows creation of false/manual coordinates if the checkerboard was not distorted
                                    ##
                                    for corner in CORNER:
                                        ManualCorners = getManuallyAdjustedCorners(number_of_corners, boardwidth=7,
                                                                                   boardheight=8, xmax=500, ymax=583)

                                elif number_of_corners == 49:
                                    for corner in CORNER:
                                        ManualCorners = getManuallyAdjustedCorners(number_of_corners, boardwidth=7,
                                                                                   boardheight=7, xmax=500, ymax=500)

                                ##
                                # using the PerspectiveDistortionMatrix function and the manuel corners
                                # the affine transformation matrix can be generated
                                ##
                                M = PerspectiveDistortionMatrix(img, CORNER, ManualCorners)

                                columns = []

                                for i in range(0, 42, 3):
                                    ##
                                    # iterate thorugh columns i.e. [x,y,liklihood,x2,y2,iklihood2...xn,yn,liklihoodn]
                                    ##
                                    coordinate_slice = escape_array[i:i + 2].T
                                    liklihood = escape_array[i + 2].T
                                    liklihood = liklihood.reshape(240, 1)
                                    ##
                                    # slice accordingly and turn 3 dimensional by z = 1
                                    ##
                                    coordinates = []
                                    for coor in coordinate_slice:
                                        xy = list(coor)
                                        xy.append(1)
                                        coordinates.append(xy)
                                    coordinate = np.array(coordinates, dtype='float32')

                                    ##
                                    # apply M matrix to coordinates
                                    ##
                                    preDiv = np.dot(M, coordinate.T)
                                    afterDiv = preDiv / preDiv[2]
                                    postLinearTransform = afterDiv[:2].T

                                    ##
                                    # restack and arrange list of coordinates back
                                    ##
                                    new_column = np.column_stack((postLinearTransform, liklihood))
                                    columns.append(new_column)

                                new_coordinates = np.column_stack(columns).T
                                np.save(save_file_name, new_coordinates, allow_pickle=True)
                                df.append({'animal+session':, 'stimulus': stim_number}, ignore_index = True)
            save_numpy_array_singleArray(path)
            return path, df

        def compute_Behavioural_Features(ProcessedData, MeanTracks, FrontMean):
            """
        this block contains the code to engineer the important features
        """
            angle_range = 3
            # define empty lists to be filled with a all repsonses
            all_directions_array = []
            all_rotations_array = []
            all_speeds_array = []
            all_elongations_array = []

            # loop through the data
            for response, mean, front in zip(ProcessedData, MeanTracks, FrontMean):

                # define empty lists to be filled with a single repsonse
                deflectionData = []
                rotationData = []
                speedData = []
                elongationData = []

                # loop through the frames
                for frame in range(frames):
                    # calc head deflections and append value to repsonse list
                    Head_deflection = calc_head_deflections(response, frame, bodyPartList, angle_range)
                    deflectionData.append(Head_deflection)

                    # calc body rotations and append value to repsonse list
                    Rotation = calc_body_rotations(mean, front, frame)
                    rotationData.append(Rotation)

                    # calc average speed and append value to repsonse list
                    meanframes = np.copy(mean.T[frame:frame + angle_range])
                    averageSpeed = find_average_speed(meanframes, len(meanframes))
                    speedData.append(averageSpeed)

                    # calc body length and append value to repsonse list
                    Elongation = get_spine_length(response, frame,
                                                  [neck, spine1, spine2, spine3, tail1, tail2, tail3, tail4, tail5])
                    elongationData.append(Elongation)

                # append full responses to lists
                all_directions_array.append(deflectionData)
                all_rotations_array.append(rotationData)
                all_speeds_array.append(speedData)
                all_elongations_array.append(elongationData)

            # convert lists to array for calibility
            HeadDef = np.array(all_directions_array)
            BodyRot = np.array(all_rotations_array)
            Speed = np.array(all_speeds_array)
            BodyElon = np.array(all_elongations_array)

            return HeadDef, BodyRot, Speed, BodyElon

        def compute_Responses(MeanTracks, Speed):
            """
        this block contains the code to compute the repsonse
        output: all reponses where the matrix: row 1 = escape, row 2 = freeze, row 3 = late return, zero matrix is a 'non-responder'
        """
            AllResponses = []
            for mean, speed in zip(MeanTracks, Speed):
                # determines reponse type
                responseType = return_response_type(mean, speed)

                # appends repsonse to array of all reponses
                AllResponses.append(responseType)

            AllResponses = np.array(AllResponses)
            return AllResponses

        def compute_Behavioural_Events(HeadDef, BodyRot, BodyElon, HeadDefThresh=1.8, BodyRotThresh=1.35,
                                       BodyElonThresh=0.8, maxmin=['max', 'max', 'min']):
            """
        this block contains the code to compute the behavioural events
        output: recombined into a matrix where row 1 = head deflection event, row 2 = rotation event, row 3 = elongation event
        """
            threshes = [HeadDefThresh, BodyRotThresh, BodyElonThresh]

            AllEvents = []

            # zip through features
            for defl, rot, elon in zip(HeadDef, BodyRot, BodyElon):

                # create a list containng the features of a single repsonse
                datas = [defl, rot, elon]
                instances = []

                # loop through features and check for events
                for data, thresh, mxmn in zip(datas, threshes, maxmin):
                    instant = create_event_occuranceVec(data, mxmn, thresh)
                    # returning a vector
                    instances.append(instant)

                AllEvents.append(instances)
                # recombined into a matrix where row 1 = head deflection event, row 2 = rotation event, row 3 = elongation event

            AllEvents = np.array(AllEvents)

            return AllEvents

        def main(csv_loc, mat_file_loc, path_to_data):
            """
        inputs:
        csv_loc - the path to the csv file containing the metadata
        mat_file_loc - the path to the Matfiles, e.g. '/research/DATA/RIGS/PS19/VIDEO_CALIBRATIONS'
        path_to_data - the path to the raw data, e.g. '/research/DATA/SUBJECTS_IoO/'
        outputs: processed data array (animals, data, frames)
        data = bodyparts, features, response, behaviour events
        """

            # turns raw data into stacked numpy array of the form (animals, bodyparts, frames)
            rawDataLoc, Mouse_escape_index = TransformDLC_output(csv_loc, mat_file_loc, path_to_data)

            # load raw data as numpy array
            rawData = np.load(rawDataLoc)

            # number of frames
            frames = 290

            # define body part slices
            nose = np.s_[0: 2]
            leftEar = np.s_[3: 5]
            rightEar = np.s_[6: 8]
            neck = np.s_[9: 11]
            spine1 = np.s_[12: 14]
            leftShoulder = np.s_[15: 17]
            rightShoulder = np.s_[18: 20]
            spine2 = np.s_[21: 23]
            spine3 = np.s_[24: 26]
            tail1 = np.s_[27: 29]
            tail2 = np.s_[30: 32]
            tail3 = np.s_[33: 35]
            tail4 = np.s_[36: 38]
            tail5 = np.s_[39: 41]

            # list containing slices
            bodyPartList = [nose, leftEar, rightEar, neck, spine1, leftShoulder, rightShoulder, spine2, spine3, tail1,
                            tail2, tail3, tail4, tail5]

            # dictionary containing indexes for liklihood values
            liklihoodIndex = ({
                'nose': 2, 'leftEar': 5, 'rightEar': 8, 'neck': 11, 'spine1': 14, 'leftShoulder': 17,
                'rightShoulder': 20,
                'spine2': 23, 'spine3': 26, 'tail1': 29, 'tail2': 32, 'tail3': 35, 'tail4': 38, 'tail5': 41
            })

            # clean up data and recapitulate
            ProcessedData = modified_ProcessRawData(rawData, bodyPartList, liklihoodIndex)

            # create associated mean tracks, i.e. vector norms
            MeanTracks = createMeanTrack(ProcessedData, bodyPartList)
            FrontpartList = [leftShoulder, rightShoulder, spine1]
            FrontMean = createMeanTrack(ProcessedData, FrontpartList)

            print('data loaded and parameters defined!')

            HeadDef, BodyRot, Speed, BodyElon = compute_Behavioural_Features(ProcessedData, MeanTracks, FrontMean)

            print('features engineered!')

            # block to determine repsonse type

            AllResponses = compute_Responses(MeanTracks, Speed)

            print('responses computed')

            # block to determine behaviour sub events

            AllEvents = compute_Behavioural_Events(HeadDef, BodyRot, BodyElon)

            print('events computed')

            seconds = 3
            entries = []
            for a, b, c, d, e, f, g, h, i, j in zip(HeadDef, BodyRot, Speed, BodyElon, [i[0] for i in AllResponses],
                                                    [i[1] for i in AllResponses], [i[2] for i in AllResponses],
                                                    [i[0] for i in AllEvents], [i[1] for i in AllEvents],
                                                    [i[2] for i in AllEvents]):
                entry = np.vstack(
                    [a[:seconds], b[:seconds], c[:seconds], d[:seconds], e[:seconds], f[:seconds], g[:seconds],
                     h[:seconds], i[:seconds], j[:seconds]])
                entries.append(entry)
            entries = np.array(entries)

            # output: a series of matrixes indexed by the index provided. the matrixs are dim (10, 240), or (features by frames)
            # row 1 = head deflections timeseries
            # row 2 = body rotations timeseries
            # row 3 = speed timeseries
            # row 4 = body elongation timeseries
            # row 5 = if and which frame an escape occured (house entry) represented by a 1 at the frame location
            # row 6 = if and which frame a freeze occured (lowest speed) represented by a 1 at the frame location
            # row 7 = if and which frame a late return occured (house entry) represented by a 1 at the frame location
            # row 8 = if and which frame a sharp head deflection occured represented by a 1 at the frame location
            # row 9 = if and which frame sharp rotation occured represented by a 1 at the frame location
            # row 10 = if and which frame an skrunching of body occured represented by a 1 at the frame location

            # how to access data
            # [i[row number] for i in data] - this is list comprehension and will let you access one of the row types
            # np.array([i[row number] for i in data]) - turining it into an array will make slicing and dicing easier
            # [np.array(index list selected)] - this is the best way to make an index to slice the data
            # np.array([i[row number] for i in data])[np.array(index list selected)] - will return you data selected and indexed by the selected index

            return entries, Mouse_escape_index