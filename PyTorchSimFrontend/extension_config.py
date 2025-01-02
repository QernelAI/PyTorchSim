import os

# Hardware info config
CONFIG_VECTOR_LANE = 128
CONFIG_SPAD_INFO = {
  "spad_vaddr" : 0xD0000000,
  "spad_paddr" : 0xD0000000,
  "spad_size" : 128 << 10
}
CONFIG_PRECISION = 4 # 32bit
CONFIG_NUM_CORES = 1
CONFIG_VLEN = 32 // CONFIG_PRECISION # 256bits / 32bits = 8 [elements]

# Tile size config
CONFIG_TILE_ROW = int(os.getenv("TORCHSIM_TILE_ROW", -1))
CONFIG_TILE_COL = int(os.getenv("TORCHSIM_TILE_COL", -1))

CONFIG_BACKENDSIM_EAGER_MODE = bool(os.getenv("BACKENDSIM_EAGER_MODE", default=False))
CONFIG_BACKENDSIM_DRYRUN = bool(int(os.getenv('BACKENDSIM_DRYRUN', default=0)))

# DUMP PATH
CONFIG_BACKEND_RESULT_PATH_KEY = os.getenv("BACKEND_RESULT_PATH_KEY")
