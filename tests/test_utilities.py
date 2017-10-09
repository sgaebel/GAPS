#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for `gaps/utilities.py`.

@author: Sebastian M. Gaebel
@email: sebastian.gaebel@ligo.org
"""

import gaps
import pytest


# Most functions in `utilities` are very platform dependent, and also
# implicitly tested through e.g. `auxiliary_source` tests. Therefore
# This faile remains very small (for now).


def test_memory_size_formatting():
    assert gaps.memory_size(int(3.2e5), SI=False, template='{:.2f} {} ({} B)') == '312.50 kiB (320000 B)'
    assert gaps.memory_size(int(3.2e5), SI=True, template='{:.2f} {} ({} B)') == '320.00 kB (320000 B)'
    assert gaps.memory_size(int(1002), SI=False, template='{:.2f} {} ({} B)') == '1002 B'
    assert gaps.memory_size(int(1002), SI=True, template='{:.2f} {} ({} B)') == '1.00 kB (1002 B)'
    assert gaps.memory_size(int(4.9563e30), SI=False, template='{:.2f} {} ({} B)') == '4099755.27 YiB (4956299999999999787158399352832 B)'
    assert gaps.memory_size(int(4.9563e30), SI=True, template='{:.2f} {} ({} B)') == '4956300.00 YB (4956299999999999787158399352832 B)'
    assert gaps.memory_size(int(12), SI=False, template='{:.2f} {} ({} B)') == '12 B'
    assert gaps.memory_size(int(12), SI=True, template='{:.2f} {} ({} B)') == '12 B'
    with pytest.raises(ValueError):
        gaps.memory_size(-1)



def test_print_devices():
    # There aren't many things to test here, so just call it for
    # all different detail levels and check that it does not fail
    gaps.print_devices()
    for lvl in range(7):
        gaps.print_devices(lvl)
    # Detail < 0 should fail
    with pytest.raises(ValueError):
        gaps.print_devices(-1)
