"""
This class loads events from dat or npy files

Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import print_function

import os

import numpy as np

from . import dat_events_tools as dat
from . import npy_events_tools as npy_format


class PSEELoader(object):
    """
    PSEELoader loads a dat or npy file and stream events
    """

    def __init__(self, datfile):
        """
        ctor
        :param datfile: binary dat or npy file
        """
        self._extension = datfile.split('.')[-1]
        assert self._extension in ["dat", "npy"], 'input file path = {}'.format(datfile)
        if self._extension == "dat":
            self._binary_format = dat
        elif self._extension == "npy":
            self._binary_format = npy_format
        self._file = open(datfile, "rb")
        self._start, self.ev_type, self._ev_size, self._size = self._binary_format.parse_header(self._file)
        assert self._ev_size != 0
        if self._extension == "dat":
            self._dtype = self._binary_format.EV_TYPE
        elif self._extension == "npy":
            self._dtype = self.ev_type
        else:
            assert False, 'unsupported extension'

        self._decode_dtype = []
        for dtype in self._dtype:
            if dtype[0] == '_':
                self._decode_dtype += [('x', 'u2'), ('y', 'u2'), ('p', 'u1')]
            else:
                self._decode_dtype.append(dtype)

        # size
        self._file.seek(0, os.SEEK_END)
        self._end = self._file.tell()
        self._ev_count = (self._end - self._start) // self._ev_size
        self.done = False
        self._file.seek(self._start)
        # If the current time is t, it means that next event that will be loaded has a
        # timestamp superior or equal to t (event with timestamp exactly t is not loaded yet)
        self.current_time = 0
        self.duration_s = self.total_time() * 1e-6

    def reset(self):
        """reset at beginning of file"""
        self._file.seek(self._start)
        self.done = False
        self.current_time = 0

    def event_count(self):
        """
        getter on event_count
        :return:
        """
        return self._ev_count

    def get_size(self):
        """"(height, width) of the imager might be (None, None)"""
        return self._size

    def __repr__(self):
        """
        prints properties
        :return:
        """
        wrd = ''
        wrd += 'PSEELoader:' + '\n'
        wrd += '-----------' + '\n'
        if self._extension == 'dat':
            wrd += 'Event Type: ' + str(self._binary_format.EV_STRING) + '\n'
        elif self._extension == 'npy':
            wrd += 'Event Type: numpy array element\n'
        wrd += 'Event Size: ' + str(self._ev_size) + ' bytes\n'
        wrd += 'Event Count: ' + str(self._ev_count) + '\n'
        wrd += 'Duration: ' + str(self.duration_s) + ' s \n'
        wrd += '-----------' + '\n'
        return wrd

    def load_n_events(self, ev_count):
        """
        load batch of n events
        :param ev_count: number of events that will be loaded
        :return: events
        Note that current time will be incremented to reach the timestamp of the first event not loaded yet
        """
        event_buffer = np.empty((ev_count + 1,), dtype=self._decode_dtype)

        pos = self._file.tell()
        count = (self._end - pos) // self._ev_size
        if ev_count >= count:
            self.done = True
            ev_count = count
            self._binary_format.stream_td_data(self._file, event_buffer, self._dtype, ev_count)
            self.current_time = event_buffer['t'][ev_count - 1] + 1
        else:
            self._binary_format.stream_td_data(self._file, event_buffer, self._dtype, ev_count + 1)
            self.current_time = event_buffer['t'][ev_count]
            self._file.seek(pos + ev_count * self._ev_size)

        return event_buffer[:ev_count]

    def load_delta_t(self, delta_t):
        """
        loads a slice of time.
        :param delta_t: (us) slice thickness
        :return: events
        Note that current time will be incremented by delta_t.
        If an event is timestamped at exactly current_time it will not be loaded.
        """
        if delta_t < 1:
            raise ValueError("load_delta_t(): delta_t must be at least 1 micro-second: {}".format(delta_t))

        if self.done or (self._file.tell() >= self._end):
            self.done = True
            return np.empty((0,), dtype=self._decode_dtype)

        final_time = self.current_time + delta_t
        tmp_time = self.current_time
        start = self._file.tell()
        pos = start
        nevs = 0
        batch = 100000
        event_buffer = []
        # data is read by buffers until enough events are read or until the end of the file
        while tmp_time < final_time and pos < self._end:
            count = (min(self._end, pos + batch * self._ev_size) - pos) // self._ev_size
            buffer = np.empty((count,), dtype=self._decode_dtype)
            self._binary_format.stream_td_data(self._file, buffer, self._dtype, count)
            tmp_time = buffer['t'][-1]
            event_buffer.append(buffer)
            nevs += count
            pos = self._file.tell()
        if tmp_time >= final_time:
            self.current_time = final_time
        else:
            self.current_time = tmp_time + 1
        assert len(event_buffer) > 0
        idx = np.searchsorted(event_buffer[-1]['t'], final_time)
        event_buffer[-1] = event_buffer[-1][:idx]
        event_buffer = np.concatenate(event_buffer)
        idx = len(event_buffer)
        self._file.seek(start + idx * self._ev_size)
        self.done = self._file.tell() >= self._end
        return event_buffer

    def seek_event(self, ev_count):
        """
        seek in the file by ev_count events
        :param ev_count: seek in the file after ev_count events
        Note that current time will be set to the timestamp of the next event.
        """
        if ev_count <= 0:
            self._file.seek(self._start)
            self.current_time = 0
        elif ev_count >= self._ev_count:
            # we put the cursor one event before and read the last event
            # which puts the file cursor at the right place
            # current_time is set to the last event timestamp + 1
            self._file.seek(self._start + (self._ev_count - 1) * self._ev_size)
            self.current_time = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0] + 1
        else:
            # we put the cursor at the *ev_count*nth event
            self._file.seek(self._start + (ev_count) * self._ev_size)
            # we read the timestamp of the following event (this change the position in the file)
            self.current_time = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0]
            # this is why we go back at the right position here
            self._file.seek(self._start + (ev_count) * self._ev_size)
        self.done = self._file.tell() >= self._end

    def seek_time(self, final_time, term_criterion=100000):
        """
        go to the time final_time inside the file. This is implemented using a binary search algorithm
        :param final_time: expected time
        :param term_cirterion: (nb event) binary search termination criterion
        it will load those events in a buffer and do a numpy searchsorted so the result is always exact
        """
        if final_time > self.total_time():
            self._file.seek(self._end)
            self.done = True
            self.current_time = self.total_time() + 1
            return

        if final_time <= 0:
            self.reset()
            return

        low = 0
        high = self._ev_count

        # binary search
        while high - low > term_criterion:
            middle = (low + high) // 2

            self.seek_event(middle)
            mid = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0]

            if mid > final_time:
                high = middle
            elif mid < final_time:
                low = middle + 1
            else:
                self.current_time = final_time
                self.done = self._file.tell() >= self._end
                return
        # we now know that it is between low and high
        self.seek_event(low)
        final_buffer = np.fromfile(self._file, dtype=self._dtype, count=high - low)['t']
        final_index = np.searchsorted(final_buffer, final_time)

        self.seek_event(low + final_index)
        self.current_time = final_time
        self.done = self._file.tell() >= self._end

    def total_time(self):
        """
        get total duration of video in mus, providing there is no overflow
        :return:
        """
        if not self._ev_count:
            return 0
        # save the state of the class
        pos = self._file.tell()
        current_time = self.current_time
        done = self.done
        # read the last event's timestamp
        self.seek_event(self._ev_count - 1)
        time = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0]
        # restore the state
        self._file.seek(pos)
        self.current_time = current_time
        self.done = done

        return time

    def __del__(self):
        self._file.close()
