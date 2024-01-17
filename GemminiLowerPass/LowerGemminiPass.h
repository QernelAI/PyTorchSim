#ifndef LOWERGEMMINIPASS_H
#define LOWERGEMMINIPASS_H

namespace llvm {
class LowerGemminiPass
    : public PassInfoMixin<LowerGemminiPass> {

public:
  LowerGemminiPass() {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);
  static bool isRequired() { return true; }
};
} // namespace llvm

/* GEMMINI DEFINITION */
#define ADDR_LEN 32
#define ROW_LEN 16
#define COL_LEN 16
#define BITMASK(x) ((1 << x) - 1)
#define ADDR_MASK BITMASK(ADDR_LEN)
#define ROW_MASK  BITMASK(ROW_LEN)
#define COL_MASK  BITMASK(COL_LEN)

#define k_CONFIG 0
#define k_MVIN2 1
#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 6
#define k_FLUSH 7

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#endif
