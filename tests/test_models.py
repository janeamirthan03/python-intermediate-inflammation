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
    """Test that daily max function works for an array where daily max appears twice"""
    from inflammation.models import daily_min

    test_input = np.array([[0, 1],
                           [0, 1],
                           [2, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_float():
    """Test that daily max function works for an array where daily max appears twice"""
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