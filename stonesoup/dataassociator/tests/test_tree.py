import datetime
import pytest
import numpy as np

from ..neighbour import (
    NearestNeighbour, GlobalNearestNeighbour, GNNWith2DAssignment)
from ..tree import DetectionKDTreeMixIn, TPRTreeMixIn
# from stonesoup.predictor.kalman import KalmanPredictor
# from stonesoup.models.transition.linear import (
#     CombinedLinearGaussianTransitionModel, ConstantVelocity)
# from stonesoup.updater.kalman import KalmanUpdater
# from stonesoup.models.measurement.linear import LinearGaussian
# from stonesoup.predictor.kalman import KalmanPredictor
# from stonesoup.measures import Mahalanobis
# from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.types.track import Track
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState


class DetectionKDTreeNN(NearestNeighbour, DetectionKDTreeMixIn):
    '''DetectionKDTreeNN from NearestNeighbour and DetectionKDTreeMixIn'''
    pass


class DetectionKDTreeGNN(GlobalNearestNeighbour, DetectionKDTreeMixIn):
    '''DetectionKDTreeGNN from GlobalNearestNeighbour and DetectionKDTreeMixIn'''
    pass


class DetectionKDTreeGNN2D(GNNWith2DAssignment, DetectionKDTreeMixIn):
    '''DetectionKDTreeGNN2D from GNNWith2DAssignment and DetectionKDTreeMixIn'''
    pass


class TPRTreeNN(NearestNeighbour, TPRTreeMixIn):
    '''TPRTreeNN from NearestNeighbour and TPRTreeMixIn'''
    pass


@pytest.fixture(params=[
    DetectionKDTreeNN, DetectionKDTreeGNN, DetectionKDTreeGNN2D])
def associator(
        request, distance_hypothesiser, probability_predictor,
        probability_updater, measurement_model):
    '''Distance associator for each KD Tree'''
    kd_trees = [DetectionKDTreeNN, DetectionKDTreeGNN, DetectionKDTreeGNN2D]
    if request.param in kd_trees:
        return request.param(distance_hypothesiser, probability_predictor, probability_updater)
    else:
        pass


@pytest.fixture(params=[DetectionKDTreeGNN2D])
def probability_associator(
        request, probability_hypothesiser, probability_predictor,
        probability_updater, measurement_model):
    '''Probability associator for each KD Tree'''
    return request.param(probability_hypothesiser, probability_predictor, probability_updater)


def test_nearest_neighbour(associator):
    '''Test method for nearest neighbour and KD tree'''
    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    t2 = Track([GaussianState(np.array([[3]]), np.array([[1]]), timestamp)])
    d1 = Detection(np.array([[2]]))
    d2 = Detection(np.array([[5]]))

    tracks = {t1, t2}
    detections = {d1, d2}

    associations = associator.associate(tracks, detections, timestamp)

    # There should be 2 associations
    assert len(associations) == 2

    # Each track should associate with a unique detection
    associated_measurements = [hypothesis.measurement
                               for hypothesis in associations.values()
                               if hypothesis.measurement]
    assert len(associated_measurements) == len(set(associated_measurements))


def test_missed_detection_nearest_neighbour(associator):
    '''Test method for nearest neighbour and KD tree'''
    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    t2 = Track([GaussianState(np.array([[3]]), np.array([[1]]), timestamp)])
    d1 = Detection(np.array([[20]]))

    tracks = {t1, t2}
    detections = {d1}

    associations = associator.associate(tracks, detections, timestamp)

    # Best hypothesis should be missed detection hypothesis
    assert all(not hypothesis.measurement
               for hypothesis in associations.values())


def test_probability_gnn(probability_associator):
    '''Test method for global nearest neighbour and KD tree'''
    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    t2 = Track([GaussianState(np.array([[3]]), np.array([[1]]), timestamp)])
    d1 = Detection(np.array([[2]]))
    d2 = Detection(np.array([[5]]))

    tracks = {t1, t2}
    detections = {d1, d2}

    associations = probability_associator.associate(
        tracks, detections, timestamp)

    # There should be 2 associations
    assert len(associations) == 2

    # Each track should associate with a unique detection
    associated_measurements = [hypothesis.measurement
                               for hypothesis in associations.values()
                               if hypothesis.measurement]
    assert len(associated_measurements) == len(set(associated_measurements))
