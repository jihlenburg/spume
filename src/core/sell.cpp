// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include "core/sell.hpp"

namespace spume {

Sell<double> sell_from_csr(const Csr& a) {
    return detail::sell_from_csr_as<double>(a);
}

Sell<double> sell_from_coo(const Coo& a) {
    return detail::sell_from_csr_as<double>(coo_to_csr(a));
}

} // namespace spume
