🔗 **Related peer-reviewed papers (full-scale context):**
- **Managerial / operational dispatch design (acceptance + batching/bundling + routing):** https://www.sciencedirect.com/science/article/pii/S0305048324001610  
  *Highlights:* Designed M1M dispatch/assignment algorithms integrating **acceptance**, **batching/bundling**, and **routing** to improve service feasibility and operational consistency under real-world constraints. Built bundling + consolidation/pricing modules, improving **revenue yield** and **capacity utilization by 3–5%**, and evaluated **profit–cost–emissions** trade-offs.
- **Methodological / demand uncertainty + decomposition:** https://www.sciencedirect.com/science/article/abs/pii/S0191261525001377  
  *Highlights:* Built **uncertainty-aware** consolidation/coordination planning for multi-stakeholder freight; improved **expected profit by 15%+ (VSS)** vs deterministic baselines under demand uncertainty using a **decomposition-based algorithm**.

*(This repository is the compact, reproducible mini-project implementation inspired by the papers above—focused on the core decision-support workflow and a clean input/output interface.)*

📦 **Dataset / benchmark instances:** https://github.com/optimalpaths/ssnd-benchmark  

---

# Dispatch Decision Support Tool

This repository provides an "input-driven decision support tool" for a dispatch / delivery organization operating time-sensitive freight flows across predefined corridors. Given an instance (demand, available services, and bundle offerings), the tool produces a "profit-optimized operating plan" and exports actionable outputs for planning and analysis.

At a high level, the tool helps answer:
- Which services should we operate?
- Which bundle offerings (discount packages) should we activate?
- When should each shipment depart (within its allowable window)?
- How do we respect capacity while managing service cost and customer-time penalties?

The solver consumes a "Excel workbook" (.xlsx) containing the full instance definition and writes results as "CSV outputs" suitable for downstream reporting, dashboards, or scenario comparison.

---

## What the tool does

Given your input workbook, the tool:

- "Assigns every shipment" to exactly one available service and departure time.
- "Selects services to operate".
- "Respects capacity" by time period for each operated service.
- "Accounts for time sensitivity", preferring departures close to each shipment’s desired pickup time (penalizes deviations).
- "Applies bundle economics", allowing fixed-cost discounts when services are selected as part of a bundle.

---

## Who this is for

- Dispatch and linehaul planning teams evaluating corridor service options
- Operations analysts comparing “what-if” scenarios (capacity changes, new services, pricing/penalty shifts)
- OR / Data Science teams integrating a solver-based planning core into a larger planning workflow

---

## Input workbook (Excel)

The tool expects "one Excel workbook" with multiple sheets. At minimum, these are used:

- `config`
- `demands`
- `services`
- `bundles`

Additional sheets may exist for completeness (nodes, arcs, etc.), but the above are the planning-critical inputs for this version.

### Input integrity rules (important)

To keep the tool predictable and production-friendly:

1) Every demand OD pair must be supported by at least one service OD pair  
2) Every service listed inside a bundle must exist in the `services` sheet  
3) IDs must be unique (`demand_id`, `service_id`, `bundle_id`)

---

## Outputs

All outputs are written as CSV files to the output directory.

### `kpi.csv`
One-row summary of performance and solution quality, typically used for scenario comparison:
- objective value
- revenue and cost breakdowns (transport, fixed, time penalty, bundle credit)
- count of opened services (fast/regular)
- number of activated bundles
- utilization and time-window adherence indicators
- runtime and optimality gap (when applicable)

### `open_services.csv`
Services selected for operation (the "service plan").

### `open_bundles.csv`
Activated bundles (discount programs used in the plan).

### `assignments.csv`
Shipment-level decisions:
- assigned service
- departure period
- pickup deviation from desired time
- whether the chosen service benefited from a bundle discount

### `service_time_loads.csv`
Operational utilization by service and time:
- total assigned volume vs capacity
- utilization percentage

---

## Notes & extensions

This version treats each service as an "end-to-end operating option". Common next steps include:
- allowing multi-leg routing / transfers
- incorporating hub handling and storage decisions directly
- adding explicit service schedules with multiple departures per period

For larger-scale network configurations, a "Benders decomposition–based" solution approach has also been developed to maintain tractability and improve solvability as the instance size grows.

