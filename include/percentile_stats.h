// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann
{
// Wall-clock nanoseconds on CLOCK_MONOTONIC.
//
// EXPLICITLY, not via Timer. Timer uses std::chrono::high_resolution_clock, which libstdc++
// aliases to system_clock -- CLOCK_REALTIME. Durations from it are fine, which is all upstream
// uses it for, but an ABSOLUTE realtime stamp cannot be joined against an eBPF trace:
// bpf_ktime_get_ns() reads CLOCK_MONOTONIC, and the two clocks differ by the boot epoch and
// drift apart under NTP. The whole value of recording per-query bounds is that join, so the
// clock is named here rather than inherited.
inline uint64_t monotonic_ns()
{
#ifdef _WINDOWS
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

struct QueryStats
{
    float total_us = 0; // total time to process query in micros
    float io_us = 0;    // total time spent in IO
    float cpu_us = 0;   // total time spent in CPU

    // The query's own [start, end] on CLOCK_MONOTONIC. total_us is a duration and cannot place
    // the query on a timeline, so an I/O trace captured alongside the search has nothing to join
    // against and every per-query attribution has to be apportioned from run-level means
    // instead. Two clock_gettime calls per query, ~40 ns against a query of milliseconds.
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;

    unsigned n_4k = 0;         // # of 4kB reads
    unsigned n_8k = 0;         // # of 8kB reads
    unsigned n_12k = 0;        // # of 12kB reads
    unsigned n_ios = 0;        // total # of IOs issued
    unsigned read_size = 0;    // total # of bytes read
    unsigned n_cmps_saved = 0; // # cmps saved
    unsigned n_cmps = 0;       // # cmps
    unsigned n_cache_hits = 0; // # cache_hits
    unsigned n_hops = 0;       // # search hops
};

// One row per query, for joining against an I/O capture of the same run.
//
// ONLY THE FIELDS THE SEARCH PATH ACTUALLY WRITES. QueryStats also declares n_8k, n_12k,
// read_size and n_cmps_saved, and nothing in PQFlashIndex assigns any of them -- they would be
// columns of zeros, which reads as a measurement rather than as an absent one. n_4k is omitted
// for the opposite reason: it is incremented on exactly the same two lines as n_ios
// (pq_flash_index.cpp:1454-1455, 1626-1627), so it is a duplicate column, not a 4 KiB subset.
//
// CSV rather than a .bin: this is a handful of columns over the query count, read once by a
// script, and save_bin's shape is one matrix of one type. The consumer converts it.
inline void save_query_stats_csv(const std::string &path, const QueryStats *stats, uint64_t len)
{
    std::ofstream out(path);
    if (!out.is_open())
    {
        std::cerr << "WARNING: could not open " << path << " -- per-query stats not written."
                  << std::endl;
        return;
    }
    out << "query_id,start_ns,end_ns,total_us,io_us,cpu_us,n_ios,n_cmps,n_cache_hits,n_hops\n";
    for (uint64_t i = 0; i < len; i++)
    {
        const QueryStats &q = stats[i];
        out << i << "," << q.start_ns << "," << q.end_ns << "," << q.total_us << "," << q.io_us << ","
            << q.cpu_us << "," << q.n_ios << "," << q.n_cmps << "," << q.n_cache_hits << "," << q.n_hops
            << "\n";
    }
    out.close();
    std::cout << "Wrote per-query stats: " << path << " (" << len << " queries)" << std::endl;
}

template <typename T>
inline T get_percentile_stats(QueryStats *stats, uint64_t len, float percentile,
                              const std::function<T(const QueryStats &)> &member_fn)
{
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++)
    {
        vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(), [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
}

template <typename T>
inline double get_mean_stats(QueryStats *stats, uint64_t len, const std::function<T(const QueryStats &)> &member_fn)
{
    double avg = 0;
    for (uint64_t i = 0; i < len; i++)
    {
        avg += (double)member_fn(stats[i]);
    }
    return avg / len;
}
} // namespace diskann
