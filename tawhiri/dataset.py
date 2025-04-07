# Copyright 2014 (C) Daniel Richman
#
# This file is part of Tawhiri.
#
# Tawhiri is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Tawhiri is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Tawhiri.  If not, see <http://www.gnu.org/licenses/>.

"""
Open a wind dataset from file by memory-mapping

Datasets downloaded from the NOAA are stored as large binary files that are
memmapped into the predictor process and thereby treated like a huge array.

:class:`Dataset` contains some utility methods to find/list datasets in a
directory, and can open (& create) dataset files.

Note: once opened, the dataset is mmaped as :attr:`Dataset.array`, which by
itself is not particularly useful.  :mod:`tawhiri.interpolate` casts it (via a
memory view) to a pointer in Cython.
"""

from collections import namedtuple
import mmap
import os
import os.path
import signal
import operator
from datetime import datetime, timedelta, timezone
import logging
import json
from pathlib import Path

logger = logging.getLogger("tawhiri.dataset")


# NB: the Sphinx autodoc output for the Dataset class has to be adjusted by
# hand. There is some duplication; see (& update) docs/code/tawhiri.rst.

class Dataset(object):
    """
    A wind dataset

    .. attribute:: array

        A :class:`mmap.mmap` object; the entire dataset mapped into memory.

    .. attribute:: ds_time

        The forecast time of this dataset (:class:`datetime.datetime`).

    .. attribute:: valid_from

        The start of validity period for this dataset (:class:`datetime.datetime`).

    .. attribute:: valid_to

        The end of validity period for this dataset (:class:`datetime.datetime`).

    """

    #: The dimensions of the dataset
    #:
    #: Note ``len(axes[i]) == shape[i]``.
    # GFS
    #shape = (2, 49, 4, 721, 1440)
    # ECMWF
    #shape = (3, 13, 4, 721, 1440)
    # MEPS
    shape = (4, 65, 4, 28, 88)

    # TODO: use the other levels too?
    # {10, 80, 100}m heightAboveGround (u, v)
    #       -- note ground, not mean sea level - would need elevation
    # 0 unknown "planetary boundry layer" (u, v) (first two records)
    # 0 surface "Planetary boundary layer height"
    # {1829, 2743, 3658} heightAboveSea (u, v)

    #: The pressure levels contained in a "pgrb2f" file from the NOAA
    pressures_pgrb2f = [10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 350, 400,
                        450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925,
                        950, 975, 1000]
    #: The pressure levels contained in a "pgrb2bf" file from the NOAA
    pressures_pgrb2bf = [1, 2, 3, 5, 7, 125, 175, 225, 275, 325, 375, 425,
                         475, 525, 575, 625, 675, 725, 775, 825, 875]
    pressures_gfs = pressures_pgrb2f + pressures_pgrb2bf
    
    # ECMWF 
    # Pressure levels:
    #  1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1 hPa
    # Model levels: 
    #  1 to 137
    # Open data levels:
    pressures_ecmwf = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    # MEPS uses model levels, where the pressure is different for each grid point,
    # but we don't really care about the pressure, so for MEPS, we don't really look at the value of the pressure level
    pressures_meps = range(0, 65)

    _axes_type = namedtuple("axes",
                ("hour", "pressure", "variable", "latitude", "longitude"))

    #: The values of the points on each axis: a 5-(named)tuple ``(hour,
    #: pressure variable, latitude, longitude)``.
    #:
    #: For example, ``axes.pressure[4]`` is ``900`` - points in
    #: cells ``dataset.array[a][4][b][c][d]`` correspond to data at 900mb.
    # GFS
    # axes = _axes_type(
    #     range(0, 2, 1),                              # hour
    #     sorted(pressures_gfs, reverse=True),         # pressure
    #     ["height", "wind_u", "wind_v", "wind_w"],    # vars
    #     [x/4.0 for x in range(-360, 360 + 1)],       # lat
    #     [x/4.0 for x in range(-720, 720)]            # lon
    # )

    # ECMWF SCDA
    # axes = _axes_type(
    #     range(0, 9, 3),                              # hour
    #     sorted(pressures_ecmwf, reverse=True),         # pressure
    #     ["height", "wind_u", "wind_v", "wind_w"],    # vars
    #     [x/4.0 for x in range(-360, 360 + 1)],       # lat
    #     [x/4.0 for x in range(-720, 720)]            # lon
    # )

    # MEPS
    axes = _axes_type(
        range(0, 4, 1),                              # hour
        sorted(pressures_meps, reverse=False),         # pressure
        ["height", "wind_u", "wind_v", "wind_w"],    # vars
        [(-472517.90625 + i * 2500.0) for i in range(0, 28)], # lat
        [(-565084.0625 + i * 2500.0) for i in range(0, 88)]   # lon
    )


    _listdir_type = namedtuple("dataset_in_row",
                ("ds_time", "suffix", "filename", "path"))

    assert shape == tuple(len(x) for x in axes)

    #: The data type of dataset elements
    element_type = 'float32'
    #: The size in bytes of `element_type`
    element_size = 4    # float32

    #: The size in bytes of the entire dataset
    size = element_size
    for _x in shape:
        size *= _x
    del _x

    #: The default location of wind data
    DEFAULT_DIRECTORY = '/srv/tawhiri-datasets'

    cached_latest = None

    # prune_latest is registered as the signal handler for SIGALRM at the
    # bottom of the file.
    @classmethod
    def prune_latest(cls, signum, stack_frame):
        cls.cached_latest = None

    @classmethod
    def _parse_filename(cls, filename):
        """Parse a dataset filename to extract validity period."""
        try:
            # Format: YYYY-MM-DDTHH-YYYY-MM-DDTHH.tawhiri
            parts = filename.split('-')
            if len(parts) != 6 or not filename.endswith('.tawhiri'):
                return None
            
            valid_from = datetime(
                int(parts[0]),  # year
                int(parts[1]),  # month
                int(parts[2][:2]),  # day
                int(parts[2][3:5]),  # hour
                tzinfo=timezone.utc
            )
            
            valid_to = datetime(
                int(parts[3]),  # year
                int(parts[4]),  # month
                int(parts[5][:2]),  # day
                int(parts[5][3:5]),  # hour
                tzinfo=timezone.utc
            )
            
            return valid_from, valid_to
        except (ValueError, IndexError):
            return None

    @classmethod
    def _find_latest_dataset(cls, directory, not_after=None, max_days_back=14):
        """Find the latest dataset in the directory structure."""
        if not_after is None:
            not_after = datetime.now(timezone.utc)
        
        print(f"Finding latest dataset in {directory} after {not_after}")

        # Start from the current date and go backwards by days
        current_date = not_after
        for _ in range(max_days_back):
            # Try to find a dataset in the current date's directory
            date_dir = current_date.strftime("%Y/%m/%d")
            full_dir = os.path.join(directory, date_dir)
            
            if os.path.exists(full_dir):
                # Look for the latest dataset in this directory
                latest_dataset = None
                latest_gen_time = None
                
                # Check each hour directory in the current day
                for hour_dir in sorted(os.listdir(full_dir), reverse=True):
                    if not hour_dir.endswith('Z'):
                        continue
                        
                    hour_path = os.path.join(full_dir, hour_dir)
                    if not os.path.isdir(hour_path):
                        continue
                        
                    # Parse generation time from directory path
                    try:
                        gen_time = datetime(
                            int(date_dir.split('/')[0]),  # year
                            int(date_dir.split('/')[1]),  # month
                            int(date_dir.split('/')[2]),  # day
                            int(hour_dir[:-1]),  # hour (remove 'Z')
                            tzinfo=timezone.utc
                        )
                    except (ValueError, IndexError):
                        continue
                        
                    print(f"Dir {hour_path}")
                    for filename in sorted(os.listdir(hour_path), reverse=True):
                        print(f"Checking {filename}")
                        if not filename.endswith('.tawhiri'):
                            continue
                            
                        result = cls._parse_filename(filename)
                        if result is None:
                            continue
                            
                        valid_from, valid_to = result
                        print(f"gen_time: {gen_time}, valid_from: {valid_from}, valid_to: {valid_to}")
                        # Check if this dataset is valid for our time
                        if valid_from <= not_after:
                            if latest_gen_time is None or gen_time > latest_gen_time:
                                latest_gen_time = gen_time
                                latest_dataset = (os.path.join(hour_path, filename), gen_time, valid_from, valid_to)
                                print("=> Possible!")
                
                if latest_dataset is not None:
                    print(f"Latest dataset: {latest_dataset}")
                    return latest_dataset
            
            # Move back one day
            current_date -= timedelta(days=1)
        
        return None

    @classmethod
    def open_latest(cls, directory=DEFAULT_DIRECTORY, persistent=False, not_after=None):
        """
        Find the most recent dataset in `directory`, and open it.

        :type directory: string
        :param directory: directory to search
        :type persistent: bool
        :param persistent: should the latest dataset be cached, and re-used?
        :type not_after: datetime
        :param not_after: if specified, only return datasets with validity period not after this datetime
        :rtype: :class:`Dataset`
        """
        result = cls._find_latest_dataset(directory, not_after)
        if result is None:
            raise ValueError(f"No valid dataset found in directory: {directory}")
            
        path, gen_time, valid_from, valid_to = result
        
        cached = cls.cached_latest
        valid = cached and \
                cached.fn == path

        if valid:
            if persistent:
                # Refresh countdown
                signal.alarm(60)
            return cls.cached_latest
        else:
            ds = Dataset(path)
            if persistent:
                # Start the countdown
                signal.alarm(60)
                # note, this creates a ref cycle.
                cls.cached_latest = ds
            return ds

    header = None
    header_size = 0

    def __init__(self, path):
        """
        Open a dataset file.

        :type path: string
        :param path: full path to the dataset file
        """
        # Parse generation time from directory path
        dir_parts = path.split('/')
        try:
            gen_time = datetime(
                int(dir_parts[-5]),  # year
                int(dir_parts[-4]),  # month
                int(dir_parts[-3]),  # day
                int(dir_parts[-2][:-1]),  # hour (remove 'Z')
                tzinfo=timezone.utc
            )
        except (ValueError, IndexError):
            raise ValueError(f"Invalid dataset path structure: {path}")

        result = self._parse_filename(os.path.basename(path))
        if result is None:
            raise ValueError(f"Invalid dataset filename: {path}")
            
        valid_from, valid_to = result

        self.directory = os.path.dirname(path)
        self.ds_time = gen_time
        self.valid_from = valid_from
        self.valid_to = valid_to
        self.fn = path

        prot = mmap.PROT_READ
        flags = mmap.MAP_SHARED
        mode = "rb"

        logger.info("Opening dataset %s %s", self.ds_time, self.fn)

        with open(self.fn, mode) as f:
            f.seek(0, 0)

            # Read header
            (self.header, self.header_size) = self.parse_header(f)

            print(f"Opened dataset: {json.dumps(self.header)} [{self.header_size}]")

            # We could offset the header here, but macOS doesn't seem to support that, so we do the offset later in MagicMemoryView instead
            self.array = mmap.mmap(f.fileno(), 0, prot=prot, flags=flags)

    @classmethod
    def parse_header(cls, f):
        # Read file in chunks until we find a NUL character
        chunk_size = 4096  # Reasonable buffer size
        json_bytes = bytearray()
        
        while True:
            chunk = f.read(chunk_size)
            if not chunk:  # End of file
                raise ValueError("No NUL terminator found in file")
            
            # Look for NUL in this chunk
            nul_index = chunk.find(b'\0')
            if nul_index != -1:
                # Found the terminator - add the bytes up to (but not including) NUL
                json_bytes.extend(chunk[:nul_index])
                # Calculate where the binary data starts (current position - remaining chunk + nul_index + 1)
                binary_start = f.tell() - len(chunk) + nul_index + 1
                break
            
            json_bytes.extend(chunk)
        
        # Parse the JSON data
        try:
            json_str = json_bytes.decode('utf-8')
            json_obj = json.loads(json_str)
            return json_obj, binary_start
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 in JSON header: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON header: {str(e)}")

    def __del__(self):
        self.close()

    def close(self):
        """
        Close the dataset

        This deletes :attr:`array`, thereby releasing (a) reference to it.
        Note that other objects may very well hold a reference to the array,
        keeping it open.

        (The file descriptor is closed as soon as the dataset is mapped.)
        """
        if hasattr(self, 'array'):
            logger.info("Closing dataset %s %s", self.ds_time, self.fn)
            del self.array


signal.signal(signal.SIGALRM, Dataset.prune_latest)
