"""
End-to-end workflow:

1) Read a single Excel workbook containing all required input sheets.
2) Validate input integrity (in particular: every demand OD pair must be covered by at least one service OD pair).
3) Formulate and solve a Gurobi MILP, including bundle-based fixed-cost discount logic.
4) Export key performance indicators (KPIs) and detailed solution outputs to CSV.

Dependencies: gurobipy, pandas, openpyxl
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Union

import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# -----------------------------
# IO + validation
# -----------------------------
REQUIRED_SHEETS = {
    "config",
    "nodes",
    "arcs",
    "time_periods",
    "demands",
    "services",
    "bundles",
    "warehousing",
    "handling",
}

REQ_COLS_DEMANDS = {
    "demand_id",
    "origin",
    "destination",
    "demand_type",
    "volume",
    "unit_revenue",
    "desired_pickup_t",
    "delivery_duration",
    "unit_penalty_per_period",
    "shift_allow",
}

REQ_COLS_SERVICES = {
    "service_id",
    "service_type",
    "origin",
    "destination",
    "duration",
    "cap_leg",
    "fixed_cost",
    "unit_cost",
}

REQ_COLS_BUNDLES = {"bundle_id", "service_ids", "discount_frac"}


def read_inputs(xlsx_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)

    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    missing = REQUIRED_SHEETS - set(sheets.keys())
    if missing:
        raise ValueError(f"Missing sheets in workbook: {sorted(missing)}")
    return sheets


def _safe_cli_args(default_xlsx: str, default_out: str) -> Tuple[str, str]:
    xlsx = default_xlsx
    out = default_out

    argv = list(sys.argv[1:])  # ignore script name

    cleaned: List[str] = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a in ("-f", "--f"):
            skip_next = True
            continue
        if a.startswith("-"):
            continue
        cleaned.append(a)

    if len(cleaned) >= 1 and Path(cleaned[0]).exists():
        xlsx = cleaned[0]
    if len(cleaned) >= 2:
        out = cleaned[1]

    return xlsx, out


def _require_columns(df: pd.DataFrame, required: Set[str], sheet_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sheet '{sheet_name}' is missing columns: {sorted(missing)}")


def validate_inputs(demands: pd.DataFrame, services: pd.DataFrame, bundles: pd.DataFrame) -> None:
    _require_columns(demands, REQ_COLS_DEMANDS, "demands")
    _require_columns(services, REQ_COLS_SERVICES, "services")
    _require_columns(bundles, REQ_COLS_BUNDLES, "bundles")

    # Unique IDs
    if demands["demand_id"].astype(str).duplicated().any():
        raise ValueError("Sheet 'demands': demand_id must be unique.")
    if services["service_id"].astype(str).duplicated().any():
        raise ValueError("Sheet 'services': service_id must be unique.")
    if bundles["bundle_id"].astype(str).duplicated().any():
        raise ValueError("Sheet 'bundles': bundle_id must be unique.")

    # OD coverage check: every demand OD must exist in service OD set
    demand_od = set(zip(demands["origin"].astype(str), demands["destination"].astype(str)))
    service_od = set(zip(services["origin"].astype(str), services["destination"].astype(str)))
    missing_od = sorted(list(demand_od - service_od))
    if missing_od:
        sample = missing_od[:10]
        raise ValueError(
            "Input invalid: some demand (origin,destination) pairs have no matching service.\n"
            f"Missing OD pairs (sample up to 10): {sample}\n"
            "Fix: add at least one service row for each missing OD pair, or regenerate inputs consistently."
        )

    # Bundle service IDs must exist
    service_ids = set(services["service_id"].astype(str))
    bad = []
    for _, r in bundles.iterrows():
        b = str(r["bundle_id"])
        ss = [x.strip() for x in str(r["service_ids"]).split(",") if x.strip()]
        missing = [s for s in ss if s not in service_ids]
        if missing:
            bad.append((b, missing))
    if bad:
        b, miss = bad[0]
        raise ValueError(f"Input invalid: bundle '{b}' references missing services: {miss}")


def build_feasible_times(
    demands: pd.DataFrame, services: pd.DataFrame, T: int
) -> Dict[Tuple[str, str], List[int]]:
    dur = dict(zip(services["service_id"].astype(str), services["duration"].astype(int)))

    feas: Dict[Tuple[str, str], List[int]] = {}
    for _, k in demands.iterrows():
        k_id = str(k["demand_id"])
        t0 = int(k["desired_pickup_t"])
        shift = int(k.get("shift_allow", 2))
        delivery_dur = int(k["delivery_duration"])
        latest_arrival = t0 + delivery_dur + shift

        t_min = max(1, t0 - shift)
        t_max = min(T, t0 + shift)

        for s_id in services["service_id"].astype(str).tolist():
            arrival_ok = []
            for t in range(t_min, t_max + 1):
                if t + int(dur[s_id]) <= latest_arrival:
                    arrival_ok.append(t)
            feas[(k_id, s_id)] = arrival_ok

    return feas


# -----------------------------
# MILP
# -----------------------------
def solve_instance(
    xlsx_path: Union[str, Path],
    out_dir: Union[str, Path] = "outputs",
    time_limit_s: int = 120,
    mip_gap: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, float]:

    sheets = read_inputs(xlsx_path)
    config = sheets["config"]
    demands = sheets["demands"].copy()
    services = sheets["services"].copy()
    bundles = sheets["bundles"].copy()

    validate_inputs(demands, services, bundles)

    # Config values
    cfg = {str(r["key"]): r["value"] for _, r in config.iterrows()}
    T = int(cfg.get("T", 14))

    # Ensure output directory exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index sets
    K = demands["demand_id"].astype(str).tolist()
    S = services["service_id"].astype(str).tolist()
    B = bundles["bundle_id"].astype(str).tolist()

    # Parameters
    vol = dict(zip(demands["demand_id"].astype(str), demands["volume"].astype(float)))
    rev = dict(zip(demands["demand_id"].astype(str), demands["unit_revenue"].astype(float)))
    t_des = dict(zip(demands["demand_id"].astype(str), demands["desired_pickup_t"].astype(int)))
    pen = dict(zip(demands["demand_id"].astype(str), demands["unit_penalty_per_period"].astype(float)))
    dtype = dict(zip(demands["demand_id"].astype(str), demands["demand_type"].astype(str)))

    s_type = dict(zip(services["service_id"].astype(str), services["service_type"].astype(str)))
    s_origin = dict(zip(services["service_id"].astype(str), services["origin"].astype(str)))
    s_dest = dict(zip(services["service_id"].astype(str), services["destination"].astype(str)))
    dur = dict(zip(services["service_id"].astype(str), services["duration"].astype(int)))
    cap = dict(zip(services["service_id"].astype(str), services["cap_leg"].astype(float)))
    fixed = dict(zip(services["service_id"].astype(str), services["fixed_cost"].astype(float)))
    unit_cost = dict(zip(services["service_id"].astype(str), services["unit_cost"].astype(float)))

    k_origin = dict(zip(demands["demand_id"].astype(str), demands["origin"].astype(str)))
    k_dest = dict(zip(demands["demand_id"].astype(str), demands["destination"].astype(str)))

    # Eligible services per demand: endpoint match
    S_k: Dict[str, List[str]] = {
        k: [s for s in S if s_origin[s] == k_origin[k] and s_dest[s] == k_dest[k]] for k in K
    }

    empty = [k for k in K if len(S_k[k]) == 0]
    if empty:
        raise ValueError(f"Internal check failed: demands with no eligible service: {empty[:10]}")

    # Bundle membership
    bundle_services: Dict[str, List[str]] = {}
    discount: Dict[str, float] = {}
    for _, r in bundles.iterrows():
        b = str(r["bundle_id"])
        ss = [x.strip() for x in str(r["service_ids"]).split(",") if x.strip()]
        bundle_services[b] = [s for s in ss if s in fixed]
        discount[b] = float(r.get("discount_frac", 0.10))

    # Feasible departure times
    feas_times = build_feasible_times(demands, services, T)

    # Build model
    m = gp.Model("M1M_with_bundles")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit_s
    m.Params.MIPGap = mip_gap

    # Variables
    y = m.addVars(S, vtype=GRB.BINARY, name="y_open_service")
    z = m.addVars(B, vtype=GRB.BINARY, name="z_select_bundle")

    w_disc: Dict[Tuple[str, str], gp.Var] = {}
    for b in B:
        for s in bundle_services.get(b, []):
            w_disc[(b, s)] = m.addVar(vtype=GRB.BINARY, name=f"w_disc[{b},{s}]")

    x: Dict[Tuple[str, str, int], gp.Var] = {}
    for k in K:
        for s in S_k[k]:
            for t in feas_times[(k, s)]:
                x[(k, s, t)] = m.addVar(vtype=GRB.BINARY, name=f"x_assign[{k},{s},{t}]")

    earl = m.addVars(K, lb=0.0, name="earliness")
    late = m.addVars(K, lb=0.0, name="lateness")

    # Constraints
    for k in K:
        m.addConstr(
            gp.quicksum(x[(k, s, t)] for s in S_k[k] for t in feas_times[(k, s)]) == 1,
            name=f"assign_once[{k}]",
        )

    for s in S:
        for t in range(1, T + 1):
            expr = gp.quicksum(vol[k] * x[(k, s, t)] for k in K if (k, s, t) in x)
            m.addConstr(expr <= cap[s] * y[s], name=f"cap[{s},{t}]")

    for s in S:
        used = gp.quicksum(x[(k, s, t)] for k in K for t in range(1, T + 1) if (k, s, t) in x)
        m.addConstr(used <= 1e6 * y[s], name=f"open_if_used[{s}]")

    for (b, s), var in w_disc.items():
        m.addConstr(var <= z[b], name=f"w_le_z[{b},{s}]")
        m.addConstr(var <= y[s], name=f"w_le_y[{b},{s}]")
    for s in S:
        m.addConstr(
            gp.quicksum(w_disc[(b, s)] for b in B if (b, s) in w_disc) <= y[s],
            name=f"at_most_one_bundle_disc[{s}]",
        )

    for k in K:
        chosen_t = gp.quicksum(t * x[(k, s, t)] for s in S_k[k] for t in feas_times[(k, s)])
        m.addConstr(earl[k] - late[k] == float(t_des[k]) - chosen_t, name=f"dev_balance[{k}]")

    # Objective
    total_revenue = gp.quicksum(rev[k] * vol[k] for k in K)
    transport_cost = gp.quicksum(unit_cost[s] * vol[k] * x[(k, s, t)] for (k, s, t) in x)
    fixed_cost = gp.quicksum(fixed[s] * y[s] for s in S)
    discount_amount = gp.quicksum(discount[b] * fixed[s] * w_disc[(b, s)] for (b, s) in w_disc)
    penalty_cost = gp.quicksum(pen[k] * vol[k] * (earl[k] + late[k]) for k in K)

    m.setObjective(total_revenue - transport_cost - fixed_cost + discount_amount - penalty_cost, GRB.MAXIMIZE)
    m.optimize()

    status = int(m.Status)
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi ended with status {status}")

    # Extract solution
    y_sol = {s: int(round(y[s].X)) for s in S}
    z_sol = {b: int(round(z[b].X)) for b in B}
    w_sol = {(b, s): int(round(var.X)) for (b, s), var in w_disc.items()}
    x_sol = [(k, s, t) for (k, s, t), var in x.items() if var.X > 0.5]

    # KPIs
    obj_val = float(m.ObjVal)
    revenue_val = float(sum(rev[k] * vol[k] for k in K))
    transport_val = float(sum(unit_cost[s] * vol[k] for (k, s, t) in x_sol))
    fixed_val = float(sum(fixed[s] * y_sol[s] for s in S))
    disc_val = float(sum(discount[b] * fixed[s] * w_sol[(b, s)] for (b, s) in w_sol if w_sol[(b, s)] == 1))
    penalty_val = float(sum(pen[k] * vol[k] * (earl[k].X + late[k].X) for k in K))

    open_regular = sum(y_sol[s] for s in S if s_type[s] == "regular")
    open_fast = sum(y_sol[s] for s in S if s_type[s] == "fast")
    open_bundles = sum(z_sol[b] for b in B)

    used_st = {(s, t) for (_, s, t) in x_sol}
    shipped_total = float(sum(vol[k] for (k, _, _) in x_sol))
    cap_total = float(sum(cap[s] for (s, _) in used_st))
    cap_usage_pct = 100.0 * shipped_total / cap_total if cap_total > 1e-9 else 0.0

    moved_outside = sum(vol[k] for (k, _, t) in x_sol if int(t) != int(t_des[k]))
    pct_outside = 100.0 * moved_outside / shipped_total if shipped_total > 1e-9 else 0.0

    moved_outside_std = sum(vol[k] for (k, _, t) in x_sol if dtype[k] == "standard" and int(t) != int(t_des[k]))
    moved_outside_urg = sum(vol[k] for (k, _, t) in x_sol if dtype[k] == "urgent" and int(t) != int(t_des[k]))
    std_total = sum(vol[k] for k in K if dtype[k] == "standard")
    urg_total = sum(vol[k] for k in K if dtype[k] == "urgent")
    pct_outside_std = 100.0 * moved_outside_std / std_total if std_total > 1e-9 else 0.0
    pct_outside_urg = 100.0 * moved_outside_urg / urg_total if urg_total > 1e-9 else 0.0

    kpi = {
        "objective": obj_val,
        "total_revenue": revenue_val,
        "transport_cost": transport_val,
        "fixed_cost": fixed_val,
        "bundle_discount_credit": disc_val,
        "time_window_penalty": penalty_val,
        "open_services_regular": open_regular,
        "open_services_fast": open_fast,
        "open_bundles": open_bundles,
        "capacity_usage_pct": cap_usage_pct,
        "pct_demand_outside_desired_pickup": pct_outside,
        "pct_standard_outside_desired_pickup": pct_outside_std,
        "pct_urgent_outside_desired_pickup": pct_outside_urg,
        "solver_status": status,
        "runtime_seconds": float(m.Runtime),
        "mip_gap": float(getattr(m, "MIPGap", 0.0)),
    }

    # Write outputs
    pd.DataFrame([kpi]).to_csv(out_dir / "kpi.csv", index=False)

    svc_rows = []
    for s in S:
        if y_sol[s] == 1:
            svc_rows.append(
                {
                    "service_id": s,
                    "service_type": s_type[s],
                    "origin": s_origin[s],
                    "destination": s_dest[s],
                    "duration": dur[s],
                    "fixed_cost": fixed[s],
                }
            )
    pd.DataFrame(svc_rows).sort_values(["service_type", "service_id"]).to_csv(
        out_dir / "open_services.csv", index=False
    )

    bun_rows = []
    for b in B:
        if z_sol[b] == 1:
            bun_rows.append(
                {
                    "bundle_id": b,
                    "service_ids": ",".join(bundle_services.get(b, [])),
                    "discount_frac": float(discount[b]),
                }
            )
    pd.DataFrame(bun_rows).to_csv(out_dir / "open_bundles.csv", index=False)

    assign_rows = []
    for (k, s, t) in x_sol:
        discounted = int(any(w_sol.get((b, s), 0) == 1 and z_sol[b] == 1 for b in B if (b, s) in w_sol))
        assign_rows.append(
            {
                "demand_id": k,
                "origin": k_origin[k],
                "destination": k_dest[k],
                "demand_type": dtype[k],
                "volume": vol[k],
                "desired_pickup_t": int(t_des[k]),
                "assigned_service_id": s,
                "depart_t": int(t),
                "pickup_dev": abs(int(t) - int(t_des[k])),
                "discounted_via_bundle": discounted,
            }
        )
    pd.DataFrame(assign_rows).sort_values(["demand_id"]).to_csv(out_dir / "assignments.csv", index=False)

    load_rows = []
    x_sol_set = set(x_sol)
    for s in S:
        if y_sol[s] == 0:
            continue
        for t in range(1, T + 1):
            load = sum(vol[k] for k in K if (k, s, t) in x_sol_set)
            if load > 1e-9:
                load_rows.append(
                    {"service_id": s, "t": t, "load": load, "cap": cap[s], "util_pct": 100.0 * load / cap[s]}
                )
    pd.DataFrame(load_rows).to_csv(out_dir / "service_time_loads.csv", index=False)

    return kpi


if __name__ == "__main__":
    DEFAULT_XLSX = "inputs.xlsx"
    DEFAULT_OUT = "outputs"

    xlsx, out = _safe_cli_args(DEFAULT_XLSX, DEFAULT_OUT)
    kpi = solve_instance(xlsx, out_dir=out, time_limit_s=120, mip_gap=1e-4, verbose=True)

    print("\nKPIs:")
    for k, v in kpi.items():
        print(f"  {k}: {v}")
