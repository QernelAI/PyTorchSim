#include "Common.h"

uint32_t generate_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}

template <typename T>
T get_config_value(json config, std::string key) {
  if (config.contains(key)) {
    return config[key];
  } else {
    throw std::runtime_error(fmt::format("Config key {} not found", key));
  }
}

SimulationConfig initialize_config(json config) {
  SimulationConfig parsed_config;
  // print json
  spdlog::info("TOGSim Config: {}", config.dump(2));

  /* Core configs */
  parsed_config.num_cores = config["num_cores"];
  if (config.contains("core_type")) {
    std::vector<std::string> core_types = config["core_type"].get<std::vector<std::string>>();

    if (core_types.size() != parsed_config.num_cores)
      throw std::runtime_error("Mismatch between num_cores and core_type list size");

    for (const auto& core_type : core_types) {
      if (core_type == "ws_mesh") {
        parsed_config.core_type.push_back(CoreType::WS_MESH);
      } else if (core_type == "stonne") {
        parsed_config.core_type.push_back(CoreType::STONNE);
      } else {
        throw std::runtime_error(fmt::format("Not implemented core type: {}", core_type));
      }
    }
  } else {
    /* Used WS as default */
    for (int i=0; i<parsed_config.num_cores; i++)
      parsed_config.core_type.push_back(CoreType::WS_MESH);
  }
  parsed_config.core_freq_mhz = config["core_freq_mhz"];
  if (config.contains("num_systolic_array_per_core"))
    parsed_config.num_systolic_array_per_core = config["num_systolic_array_per_core"];
  if (config.contains("num_stonne_per_core"))
    parsed_config.num_stonne_per_core = config["num_stonne_per_core"];
   if (config.contains("num_stonne_port"))
    parsed_config.num_stonne_port = config["num_stonne_port"];
  parsed_config.core_print_interval = get_config_value<uint32_t>(config, "core_stats_print_period_cycles");

  /* Stonne config */ 
  if (config.contains("stonne_config_path"))
    parsed_config.stonne_config_path = config["stonne_config_path"];

  /* DRAM config */
  if ((std::string)config["dram_type"] == "simple")
    parsed_config.dram_type = DramType::SIMPLE;
  else if ((std::string)config["dram_type"] == "ramulator")
    parsed_config.dram_type = DramType::RAMULATOR1;
  else if ((std::string)config["dram_type"] == "ramulator2")
    parsed_config.dram_type = DramType::RAMULATOR2;
  else
    throw std::runtime_error(fmt::format("Not implemented dram type {} ",
                                         (std::string)config["dram_type"]));
  parsed_config.dram_freq_mhz = config["dram_freq_mhz"];
  if (config.contains("dram_latency"))
    parsed_config.dram_latency = config["dram_latency"];
  if (config.contains("ramulator_config_path"))
    parsed_config.dram_config_path = config["ramulator_config_path"];
  parsed_config.dram_channels = config["dram_channels"];
  if (config.contains("dram_req_size_byte"))
    parsed_config.dram_req_size = config["dram_req_size_byte"];
  if (config.contains("dram_stats_print_period_cycles"))
    parsed_config.dram_print_interval = config["dram_stats_print_period_cycles"];
  if(config.contains("dram_num_burst_length"))
    parsed_config.dram_nbl = config["dram_num_burst_length"];
  if (config.contains("dram_num_partitions")) {
    parsed_config.dram_num_partitions = config["dram_num_partitions"];
    if (parsed_config.dram_channels % parsed_config.dram_num_partitions != 0) {
      throw std::runtime_error("[Config] DRAM channels must be divisible by dram_num_partitions");
    }
  }
  parsed_config.dram_channels_per_partitions =
    parsed_config.dram_channels / parsed_config.dram_num_partitions;


   /* L2D config */
  if (config.contains("l2d_type")) {
    if ((std::string)config["l2d_type"] == "nocache")
      parsed_config.l2d_type = L2CacheType::NOCACHE;
    else if ((std::string)config["l2d_type"] == "datacache")
      parsed_config.l2d_type = L2CacheType::DATACACHE;
    else
      throw std::runtime_error(fmt::format("Not implemented l2 cache type {} ",
                                          (std::string)config["l2d_type"]));
  } else {
    parsed_config.l2d_type = L2CacheType::NOCACHE;
  }

  if (config.contains("l2d_config"))
    parsed_config.l2d_config_str = config["l2d_config"];
  if (config.contains("l2d_hit_latency"))
    parsed_config.l2d_config_str = config["l2d_hit_latency"];

  /* Icnt config */
  if ((std::string)config["icnt_type"] == "simple")
    parsed_config.icnt_type = IcntType::SIMPLE;
  else if ((std::string)config["icnt_type"] == "booksim2")
    parsed_config.icnt_type = IcntType::BOOKSIM2;
  else
    throw std::runtime_error(fmt::format("Not implemented icnt type {} ",
                                         (std::string)config["icnt_type"]));
  parsed_config.icnt_freq_mhz = config["icnt_freq_mhz"];
  if (config.contains("icnt_latency_cycles"))
    parsed_config.icnt_latency = config["icnt_latency_cycles"];
  if (config.contains("booksim_config_path"))
    parsed_config.icnt_config_path = config["booksim_config_path"];
  if (config.contains("icnt_stats_print_period_cycles"))
    parsed_config.icnt_stats_print_period_cycles = config["icnt_stats_print_period_cycles"];
  if (config.contains("icnt_injection_ports_per_core"))
    parsed_config.icnt_injection_ports_per_core = config["icnt_injection_ports_per_core"];

  if (config.contains("scheduler"))
    parsed_config.scheduler_type = config["scheduler"];
  if (config.contains("num_partition"))
    parsed_config.num_partition = config["num_partition"];
  if (config.contains("partition")) {
    for (int i=0; i<parsed_config.num_cores; i++) {
      std::string core_partition = "core_" + std::to_string(i);
      uint32_t partition_id = uint32_t(config["partition"][core_partition]);
      parsed_config.partiton_map[i] = partition_id;
      spdlog::info("[Config/Core] CPU {}: Partition {}", i, partition_id);
    }
  } else {
    /* Default: all partition 0 */
    for (int i=0; i<parsed_config.num_cores; i++) {
      parsed_config.partiton_map[i] = 0;
      spdlog::info("[Config/Core] CPU {}: Partition {}", i, 0);
    }
  }

  /* Local DRAM / DSP config */
  if (config.contains("local_dram_mode"))
    parsed_config.local_dram_mode = config["local_dram_mode"];
  if (config.contains("local_dram_latency_ns"))
    parsed_config.local_dram_latency_ns = config["local_dram_latency_ns"];
  if (config.contains("dsp_core_id"))
    parsed_config.dsp_core_id = config["dsp_core_id"];
  if (config.contains("dsp_sram_latency_ns"))
    parsed_config.dsp_sram_latency_ns = config["dsp_sram_latency_ns"];
  if (config.contains("dsp_compute_scale"))
    parsed_config.dsp_compute_scale = config["dsp_compute_scale"];
  if (parsed_config.local_dram_mode)
    spdlog::info("[Config] Local DRAM mode enabled, DRAM latency: {}ns, DSP core: {}, DSP SRAM latency: {}ns, DSP compute scale: {}",
                 parsed_config.local_dram_latency_ns, parsed_config.dsp_core_id, parsed_config.dsp_sram_latency_ns, parsed_config.dsp_compute_scale);

  /* Multi-group Q32:DSP config */
  if (config.contains("q32_groups")) {
    for (auto& group_json : config["q32_groups"]) {
      SimulationConfig::Q32Group group;
      group.dsp_core = group_json["dsp_core"];
      group.q32_cores = group_json["q32_cores"].get<std::vector<int>>();
      parsed_config.q32_groups.push_back(group);
      for (int qid : group.q32_cores)
        parsed_config.core_to_dsp[qid] = group.dsp_core;
      parsed_config.core_to_dsp[group.dsp_core] = group.dsp_core;
      spdlog::info("[Config] Q32 group: DSP core {}, Q32 cores [{}]",
                   group.dsp_core, fmt::join(group.q32_cores, ", "));
    }
    if (!parsed_config.q32_groups.empty())
      parsed_config.dsp_core_id = parsed_config.q32_groups[0].dsp_core;
  } else if (parsed_config.dsp_core_id >= 0) {
    // Backward compat: single group from scalar dsp_core_id
    SimulationConfig::Q32Group group;
    group.dsp_core = parsed_config.dsp_core_id;
    for (int i = 0; i < (int)parsed_config.num_cores; i++)
      if (i != parsed_config.dsp_core_id)
        group.q32_cores.push_back(i);
    parsed_config.q32_groups.push_back(group);
    for (int i = 0; i < (int)parsed_config.num_cores; i++)
      parsed_config.core_to_dsp[i] = parsed_config.dsp_core_id;
  }

  return parsed_config;
}
