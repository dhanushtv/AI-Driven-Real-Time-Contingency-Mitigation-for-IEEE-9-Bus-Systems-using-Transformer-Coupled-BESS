import pandapower as pp
import numpy as np

def create_ieee9_network():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=138.0, name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=138.0, name="Bus 2")
    b3 = pp.create_bus(net, vn_kv=138.0, name="Bus 3")
    b4 = pp.create_bus(net, vn_kv=138.0, name="Bus 4")
    b5 = pp.create_bus(net, vn_kv=138.0, name="Bus 5")
    b6 = pp.create_bus(net, vn_kv=138.0, name="Bus 6")
    b7 = pp.create_bus(net, vn_kv=138.0, name="Bus 7")
    b8 = pp.create_bus(net, vn_kv=138.0, name="Bus 8")
    b9 = pp.create_bus(net, vn_kv=138.0, name="Bus 9")
    
    pp.create_ext_grid(net, bus=b1, vm_pu=1.0)
    pp.create_gen(net, bus=b2, p_mw=163.0, vm_pu=1.02)
    pp.create_gen(net, bus=b3, p_mw=85.0, vm_pu=1.02)
    
    pp.create_load(net, bus=b5, p_mw=125.0, q_mvar=50.0)
    pp.create_load(net, bus=b8, p_mw=100.0, q_mvar=35.0)
    pp.create_load(net, bus=b6, p_mw=90.0, q_mvar=30.0)
    
    pp.create_line_from_parameters(net, from_bus=b4, to_bus=b5, length_km=10, r_ohm_per_km=0.01, x_ohm_per_km=0.085, c_nf_per_km=0, max_i_ka=1.2, name="Line 4-5")
    pp.create_line_from_parameters(net, from_bus=b5, to_bus=b7, length_km=10, r_ohm_per_km=0.032, x_ohm_per_km=0.161, c_nf_per_km=0, max_i_ka=1.2, name="Line 5-7")
    pp.create_line_from_parameters(net, from_bus=b7, to_bus=b8, length_km=10, r_ohm_per_km=0.0085, x_ohm_per_km=0.072, c_nf_per_km=0, max_i_ka=1.2, name="Line 7-8")
    pp.create_line_from_parameters(net, from_bus=b8, to_bus=b9, length_km=10, r_ohm_per_km=0.0119, x_ohm_per_km=0.1008, c_nf_per_km=0, max_i_ka=1.2, name="Line 8-9")
    pp.create_line_from_parameters(net, from_bus=b9, to_bus=b6, length_km=10, r_ohm_per_km=0.039, x_ohm_per_km=0.17, c_nf_per_km=0, max_i_ka=1.2, name="Line 9-6")
    pp.create_line_from_parameters(net, from_bus=b6, to_bus=b4, length_km=10, r_ohm_per_km=0.0119, x_ohm_per_km=0.1008, c_nf_per_km=0, max_i_ka=1.2, name="Line 6-4")
    
    bess_mv_bus = pp.create_bus(net, vn_kv=6.6, name="BESS_6.6kV_Bus")
    pp.create_transformer_from_parameters(
        net, hv_bus=b7, lv_bus=bess_mv_bus, sn_mva=100.0,
        vn_hv_kv=138.0, vn_lv_kv=6.6, vk_percent=10.0, vkr_percent=0.5,
        pfe_kw=0.0, i0_percent=0.0, name="BESS_Transformer"
    )
    pp.create_sgen(net, bus=bess_mv_bus, p_mw=0.0, q_mvar=0.0, name="BESS_Unit")
    return net

def run_adaptive_n1_bess_control(net, tripped_line_idx, target_loading_pct=95.0):
    print(f"\n--- SIMULATING N-1 CONTINGENCY: LINE {tripped_line_idx} TRIP ---")
    net.line.at[tripped_line_idx, 'in_service'] = False
    pp.runpp(net)
    
    overloaded = net.res_line[net.res_line.loading_percent > 100.0]
    if overloaded.empty:
        print("No lines overloaded after contingency.")
        return
        
    critical_line = overloaded.loading_percent.idxmax()
    initial_loading = net.res_line.at[critical_line, 'loading_percent']
    print(f"CRITICAL: Line {critical_line} is overloaded at {initial_loading:.2f}%")
    
    p_flow_mw = abs(net.res_line.at[critical_line, 'p_from_mw'])
    i_ka_limit = net.line.at[critical_line, 'max_i_ka']
    vn_kv = net.bus.at[net.line.at[critical_line, 'from_bus'], 'vn_kv']
    mva_limit = i_ka_limit * vn_kv * np.sqrt(3)
    
    bess_sgen_idx = net.sgen[net.sgen.name == "BESS_Unit"].index[0]
    delta_p = 1.0  
    net.sgen.at[bess_sgen_idx, 'p_mw'] += delta_p
    pp.runpp(net)
    
    p_flow_perturbed = abs(net.res_line.at[critical_line, 'p_from_mw'])
    ptdf = (p_flow_perturbed - p_flow_mw) / delta_p
    net.sgen.at[bess_sgen_idx, 'p_mw'] -= delta_p
    
    for _ in range(3):
        current_loading = net.res_line.at[critical_line, 'loading_percent']
        if abs(current_loading - target_loading_pct) < 0.2:
            break
        p_flow_mw = abs(net.res_line.at[critical_line, 'p_from_mw'])
        target_flow_mw = (target_loading_pct / 100.0) * mva_limit
        mw_reduction_needed = p_flow_mw - target_flow_mw
        delta_injection = mw_reduction_needed / abs(ptdf)
        net.sgen.at[bess_sgen_idx, 'p_mw'] += delta_injection
        pp.runpp(net)

    final_loading = net.res_line.at[critical_line, 'loading_percent']
    final_mw = net.sgen.at[bess_sgen_idx, 'p_mw']
    print(f"SUCCESS: BESS injected {final_mw:.2f} MW at Bus 7 via 6.6kV/138kV Transformer.")
    print(f"Line {critical_line} loading dropped from {initial_loading:.2f}% -> {final_loading:.2f}%")
if __name__ == "__main__":
    network = create_ieee9_network()
    
    # Scale loads to simulate peak demand (115%)
    network.load['p_mw'] *= 1.15
    network.load['q_mvar'] *= 1.15
    
    # Now run the contingency on Line 1
    run_adaptive_n1_bess_control(net=network, tripped_line_idx=1, target_loading_pct=95.0)