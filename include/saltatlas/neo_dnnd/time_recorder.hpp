// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <chrono>
#include <iostream>
#include <stack>
#include <string_view>
#include <unordered_map>

#include <saltatlas/dnnd/detail/utilities/time.hpp>

namespace saltatlas {

class time_recorder_base {
 public:
  virtual ~time_recorder_base() = default;

  /// \brief Start a timer.
  virtual void start(const std::string& name) = 0;

  /// \brief Stop the most recent timer.
  /// Return the elapsed time in seconds.
  virtual double stop() = 0;

  /// \brief Stop the most recent timer and report the elapsed time.
  template <typename stream_type>
  void stop_and_report(stream_type& ostream) {
    const auto name        = get_current_name();
    const auto elapsed_sec = stop();
    ostream << name << " took (s):\t" << elapsed_sec << std::endl;
  }

  /// \brief Reset internal data.
  virtual void reset() = 0;

  /// \brief Turn on the profiler.
  virtual void turn_on() = 0;

  /// \brief Turn off the profiler.
  virtual void turn_off() = 0;

  /// \brief Get the name of the current timer.
  virtual const std::string& get_current_name() const = 0;
};

class time_recorder : public time_recorder_base {
 public:
  struct time_entry {
    std::string name{};
    double      t{0.0};
    std::size_t depth{0};
  };

  time_recorder() = default;

  void turn_on() override { m_record = true; }

  void turn_off() override { m_record = false; }

  void start(const std::string& name) override {
    m_name_stack.push(name);
    m_clock_stack.push(dndetail::launch_timer());
    if (!priv_contains(name)) {
      m_time_table.emplace_back(
          time_entry{.name = name, .t = 0.0, .depth = m_name_stack.size() - 1});
    }
  }

  double stop() override {
    if (num_running_timers() == 0) {
      throw std::runtime_error("No running timers.");
      return 0.0;
    }

    assert(m_clock_stack.size() > 0);
    assert(m_name_stack.size() > 0);

    const auto elapsed_sec = dndetail::elapsed_time_sec(m_clock_stack.top());
    m_clock_stack.pop();

    const auto name = m_name_stack.top();
    priv_find(name).t += elapsed_sec;
    m_name_stack.pop();

    return elapsed_sec;
  }

  void reset() override {
    for (auto& entry : m_time_table) {
      reset(entry.name);
    }
  }

  void reset(const std::string& name) { priv_find(name).t = 0.0; }

  const std::string& get_current_name() const { return m_name_stack.top(); }

  const std::vector<time_entry>& get_time_table() const { return m_time_table; }

  template <typename stream_type>
  void print(stream_type& ostream) const {
    for (auto& entry : m_time_table) {
      for (std::size_t i = 0; i < entry.depth; ++i) {
        ostream << "-- ";
      }
      ostream << entry.name << ":\t" << entry.t << std::endl;
    }
  }

  std::size_t num_running_timers() const { return m_clock_stack.size(); }

 private:
  bool priv_contains(const std::string& name) const {
    return std::find_if(m_time_table.begin(), m_time_table.end(),
                        [&name](const auto& entry) {
                          return entry.name == name;
                        }) != m_time_table.end();
  }

  const time_entry& priv_find(const std::string& name) const {
    const auto itr =
        std::find_if(m_time_table.begin(), m_time_table.end(),
                     [&name](const auto& entry) { return entry.name == name; });
    assert(itr != m_time_table.end());
    return *itr;
  }

  time_entry& priv_find(const std::string& name) {
    auto itr =
        std::find_if(m_time_table.begin(), m_time_table.end(),
                     [&name](const auto& entry) { return entry.name == name; });
    assert(itr != m_time_table.end());
    return *itr;
  }

  bool                                                       m_record{true};
  std::vector<time_entry>                                    m_time_table;
  std::stack<std::chrono::high_resolution_clock::time_point> m_clock_stack;
  std::stack<std::string>                                    m_name_stack;
};

}  // namespace saltatlas::neo_dnnd