#pragma once

#warning \
    "saltatlas/dnnd/dnnd_advanced.hpp is deprecated and will be removed in future releases. Please use saltatlas/dnnd/dnnd_adv.hpp instead."

#include <saltatlas/dnnd/dnnd_adv.hpp>

namespace saltatlas {

/// \warning This alias is deprecated and will be removed in future releases.
/// Please use `dnnd_adv` instead.
template <typename Id       = uint64_t,
          typename Point    = saltatlas::pm_feature_vector<double>,
          typename Distance = double, typename IdHash = std::hash<Id>>
using dnnd = dnnd_adv<Id, Point, Distance, IdHash>;

}  // namespace saltatlas