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

  /// \brief Start a timer with the given name. If a timer with the same name
  /// already exists, it will be reused (i.e., the elapsed time will be added
  /// to the existing entry).
  void start(const std::string& name) override {
    if (!priv_valid_name(name)) {
      std::cerr << "Warning: invalid timer name: " << name
                << ". Timer names cannot contain '$' or '#' characters."
                << std::endl;
      return;
    }

    m_clock_stack.push(dndetail::launch_timer());
    m_name_stack.push(name);
    priv_find_or_create(priv_get_stacked_name(name));
  }

  /// \brief Stop the most recent timer and report the elapsed time.
  /// Return the elapsed time in seconds.
  double stop() override {
    if (num_running_timers() == 0) {
      throw std::runtime_error("No running timers.");
      return 0.0;
    }

    assert(m_clock_stack.size() > 0);
    assert(m_name_stack.size() > 0);

    const auto elapsed_sec = dndetail::elapsed_time_sec(m_clock_stack.top());
    m_clock_stack.pop();

    const auto name         = m_name_stack.top();
    const auto stacked_name = priv_get_stacked_name(name);
    priv_find(stacked_name).t += elapsed_sec;
    m_name_stack.pop();

    return elapsed_sec;
  }

  void reset() override {
    for (auto& entry : m_time_table) {
      entry.t = 0.0;
    }
  }

  const std::string& get_current_name() const override {
    return m_name_stack.top();
  }

  std::vector<time_entry> get_time_table() const {
    std::vector<time_entry> copy = m_time_table;
    for (auto& entry : copy) {
      entry.name = priv_get_original_name(entry.name);
    }
    return copy;
  }

  template <typename stream_type>
  void print(stream_type& ostream) const {
    for (auto& entry : m_time_table) {
      for (std::size_t i = 0; i < entry.depth; ++i) {
        ostream << "-- ";
      }
      ostream << priv_get_original_name(entry.name) << ":\t" << entry.t
              << std::endl;
    }
  }

  std::size_t num_running_timers() const { return m_clock_stack.size(); }

 private:
  // check if the name is valid (i.e., it does not contain '$' and '#', which is
  // reserved for internal use)
  bool priv_valid_name(const std::string& name) const {
    if (name.find('$') != std::string::npos ||
        name.find('#') != std::string::npos) {
      return false;
    }
    return true;
  }

  std::string priv_get_stacked_name(const std::string& name) const {
    std::string stacked_name;
    // add all stacked name from bottom to top, separated by "->"
    std::stack<std::string>  temp_stack = m_name_stack;
    std::vector<std::string> names;
    while (!temp_stack.empty()) {
      names.push_back(temp_stack.top());
      temp_stack.pop();
    }
    std::reverse(names.begin(), names.end());
    for (const auto& n : names) {
      stacked_name += n + "$";
    }
    stacked_name += '#';
    stacked_name += name;
    return stacked_name;
  }

  std::string priv_get_original_name(const std::string& stacked_name) const {
    const auto pos = stacked_name.find('#');
    if (pos == std::string::npos) {
      return stacked_name;
    }
    return stacked_name.substr(pos + 1);
  }

  void priv_find_or_create(const std::string& stacked_name) {
    if (!priv_contains(stacked_name)) {
      m_time_table.emplace_back(time_entry{
          .name = stacked_name, .t = 0.0, .depth = m_name_stack.size() - 1});
    }
  }

  bool priv_contains(const std::string& stacked_name) const {
    return std::find_if(m_time_table.begin(), m_time_table.end(),
                        [&stacked_name](const auto& entry) {
                          return entry.name == stacked_name;
                        }) != m_time_table.end();
  }

  time_entry& priv_find(const std::string& stacked_name) {
    auto itr = std::find_if(m_time_table.begin(), m_time_table.end(),
                            [&stacked_name](const auto& entry) {
                              return entry.name == stacked_name;
                            });
    assert(itr != m_time_table.end());
    return *itr;
  }

  bool                                                       m_record{true};
  std::vector<time_entry>                                    m_time_table;
  std::stack<std::chrono::high_resolution_clock::time_point> m_clock_stack;
  std::stack<std::string>                                    m_name_stack;
};

}  // namespace saltatlas