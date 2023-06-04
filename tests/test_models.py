"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_max_recurring_integer():
    """Test that daily max function works for an array where daily max appears twice"""
    from inflammation.models import daily_max

    test_input = np.array([[0, 1],
                          [0, 1],
                          [2, 0]])
    test_result = np.array([2, 1])

    #Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_max_float():
    """Test that daily max function works for an array where the daily max has float"""
    from inflammation.models import daily_max

    test_input = np.array([[0, 0.5],
                           [0, 0],
                           [0, 0.6]])
    test_result = np.array([0, 0.6])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_min_zeros():
    """Test that daily min function works for an array where daily min appears twice"""
    from inflammation.models import daily_min

    test_input = np.array([[0, 1],
                           [0, 1],
                           [2, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_float():
    """Test that daily min function works for an array where daily min is a float"""
    from inflammation.models import daily_min

    test_input = np.array([[0, 1.5],
                           [0, 1.3],
                           [0.1, 1.2]])
    test_result = np.array([0, 1.2])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)

def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])

@pytest.mark.parametrize(
    'test, expected',
    [
        ([[0, 1], [0, 1], [2, 0]], [2, 1]),
        ([[0, 0.5], [0, 0.6], [0, 0]], [0, 0.6]),

    ])
def test_daily_max(test, expected):
    """Test daily max function works for array of recurring integers and floats"""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    'test, expected',
    [
        ([[0, 1], [0, 1], [0, 0]], [0, 0]),
        ([[0, 0.5], [0, 0.6], [0, 0]], [0, 0.0]),

    ])
def test_daily_min(test, expected):
    """Test daily min function works for array of zeros and floats"""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]])
    ])
def test_patient_normalise(test, expected):
    """Test normalisation works for arrays of one and positive integers.
       Assumption that test accuracy of two decimal places is sufficient."""
    from inflammation.models import patient_normalise
    npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]])
    ])
def test_patient_normalise(test, expected):
    """Test normalisation works for arrays of one and positive integers.
       Assumption that test accuracy of two decimal places is sufficient."""
    from inflammation.models import patient_normalise
    npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)