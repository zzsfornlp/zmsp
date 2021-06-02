//

// Modified high-order nonproj parser from AD3's parsing example and TurboParser
// upon https://github.com/andre-martins/AD3/commit/22131c7457614dd159546500cd1a0fd8cdf2d282
// upon https://github.com/andre-martins/TurboParser/commit/a87b8e45694c18b826bb3c42e8344bd32928007d

#include "ad3/FactorGraph.h"
#include "ad3/Utils.h"
#include "utils.h"
#include "FactorTree.h"
#include "FactorHead.h"
#include <fstream>

// deubg on windows
#if defined(_WIN32)
//#define MY_DEBUG
//#define USE_PSDD
#endif

#ifndef MY_DEBUG
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif // MY_DEBUG

// =====
void parse_with_packs(const int slen, const bool projective,
  const O1Pack* po1, const O2SibPack* po2sib, const O2gPack* po2g, const O3gsibPack* po3gsib,
  vector<int>* output_heads, double* output_value
  ){
  // Start
  // Variables of the factor graph.
  vector<AD3::BinaryVariable*> variables;
  // Create factor graph.
  AD3::FactorGraph *factor_graph = new AD3::FactorGraph;

  // todo(note): po1 must exist for gh! And po1[0,0]=false, and disallow 0 at m or s!
  // get basic o1 info: [m, h]
  const vector<int>& o1_masks = po1->masks;
  const vector<double>& o1_scores = po1->scores;
  // sparse repr
  vector<int> so1_hs;
  vector<int> so1_ms;
  vector<double> so1_scores;

  // Build arc basic variables
  // [h, m] -> index in the flattened all valid arcs (basic variables)
  vector<vector<int> > index_arcs = vector<vector<int> >(slen, vector<int>(slen, -1));  // [h, m] for the algorithm
  int arc_idx = 0;
  int tmp_cur_idx = 0;  // skip mod==0 as root
  for(int m = 0; m < slen; m++){
    for(int h = 0; h < slen; h++){
      if(o1_masks[tmp_cur_idx]){
        double s = o1_scores[tmp_cur_idx];
        AD3::BinaryVariable* v = factor_graph->CreateBinaryVariable();
#ifdef FACTOR_TREE_ADDITIONAL_SCORE
        // todo(WARN): no set potential
        v->SetLogPotential(0.);
#else
        // todo(WARN): set potential here, which will be divided by num-links per var
        v->SetLogPotential(s);
#endif
        variables.push_back(v);
        // idx in all valid arcs (basic variables)
        index_arcs[h][m] = arc_idx;
        arc_idx++;
        //
        so1_hs.push_back(h);
        so1_ms.push_back(m);
        so1_scores.push_back(s);
      }
      tmp_cur_idx++;
    }
  }

  // Build tree factor
  AD3::FactorTree *tree_factor = new AD3::FactorTree;
  tree_factor->Initialize(projective, slen, &so1_hs, &so1_ms);
  vector<AD3::BinaryVariable*> local_variables = variables;  // copy
#ifdef FACTOR_TREE_ADDITIONAL_SCORE
  // todo(note): extra o1 scores!
  tree_factor->SetAdditionalLogPotentials(so1_scores);
#endif
  factor_graph->DeclareFactor(tree_factor, local_variables, true);  // put variables in

  // Collect high-order parts
  // the ordering is [o2sib, o2g, o3gsib]
  // the main purpose is to get a compact inputs for each head-automaton (split into different heads)
  // First collect indexes into the *Parts

  // o2sib: [h,m,s]: "m==s" means start child
  vector<vector<int> > po2sib_left_idxes(slen);
  vector<vector<int> > po2sib_right_idxes(slen);
  vector<vector<double> > po2sib_left_scores(slen);
  vector<vector<double> > po2sib_right_scores(slen);
  if(po2sib){
    for(int r = 0; r < po2sib->size(); r++){
      // get one entry
      int cur_h = po2sib->idxes_h[r];
      int cur_m = po2sib->idxes_m[r];
      int cur_s = po2sib->idxes_s[r];
      double cur_score = po2sib->scores[r];
      if(cur_m == 0 || cur_s == 0)
        continue;
      // left or right
      if(cur_h > cur_m){  // left
        if(cur_s >= cur_m && cur_s < cur_h){  // only get valid ones
          po2sib_left_idxes[cur_h].push_back(r);
          po2sib_left_scores[cur_h].push_back(cur_score);
        }
      }
      else{  // right
        if(cur_s <= cur_m && cur_s > cur_h){
          po2sib_right_idxes[cur_h].push_back(r);
          po2sib_right_scores[cur_h].push_back(cur_score);
        }
      }
    }
  }

  // o2g: [g,h,m]: when "h==0" => "g==0"
  vector<vector<int> > po2g_left_idxes(slen);
  vector<vector<int> > po2g_right_idxes(slen);
  vector<vector<double> > po2g_left_scores(slen);
  vector<vector<double> > po2g_right_scores(slen);
  if(po2g){
    for(int r = 0; r < po2g->size(); r++){
      // get one entry
      int cur_h = po2g->idxes_h[r];
      int cur_m = po2g->idxes_m[r];
      int cur_g = po2g->idxes_g[r];
      double cur_score = po2g->scores[r];
      if(cur_m == 0 || cur_m == cur_g)
        continue;
      // left or right
      if(cur_h > cur_m){  // left
        po2g_left_idxes[cur_h].push_back(r);
        po2g_left_scores[cur_h].push_back(cur_score);
      }
      else{  // right
        po2g_right_idxes[cur_h].push_back(r);
        po2g_right_scores[cur_h].push_back(cur_score);
      }
    }
  }

  // o3gsib: [g,h,m,s] ... (both)
  vector<vector<int> > po3gsib_left_idxes(slen);
  vector<vector<int> > po3gsib_right_idxes(slen);
  vector<vector<double> > po3gsib_left_scores(slen);
  vector<vector<double> > po3gsib_right_scores(slen);
  if(po3gsib){
    for(int r = 0; r < po3gsib->size(); r++){
      // get one entry
      int cur_h = po3gsib->idxes_h[r];
      int cur_m = po3gsib->idxes_m[r];
      int cur_s = po3gsib->idxes_s[r];
      int cur_g = po3gsib->idxes_g[r];
      double cur_score = po3gsib->scores[r];
      if(cur_m == 0 || cur_s == 0 || cur_m == cur_g || cur_s == cur_g)
        continue;
      // left or right
      if(cur_h > cur_m){  // left
        if(cur_s >= cur_m && cur_s < cur_h){  // only get valid ones
          po3gsib_left_idxes[cur_h].push_back(r);
          po3gsib_left_scores[cur_h].push_back(cur_score);
        }
      }
      else{  // right
        if(cur_s <= cur_m && cur_s > cur_h){
          po3gsib_right_idxes[cur_h].push_back(r);
          po3gsib_right_scores[cur_h].push_back(cur_score);
        }
      }
    }
  }

  // Build Head Automaton (go through each head and creat left/right)
  // TODO(+N): [0,0] as both head and mod --> currently seem fine?
  for(int cur_head = 0; cur_head < slen; cur_head++){
    // collect gs
    vector<int> gs;
    vector<AD3::BinaryVariable*> gs_variables;
    for(int g = 0; g < slen; g++){
      int index_in_valids = index_arcs[g][cur_head];
      if(index_in_valids >= 0){
        gs.push_back(g);
        gs_variables.push_back(variables[index_in_valids]);
      }
    }
    // collect ms-left and build
    if(cur_head > 0){
      // collect
      vector<int> ms_left;
      vector<AD3::BinaryVariable*> left_variables = gs_variables;
      for(int m = cur_head - 1; m > 0; m--){
        int index_in_valids = index_arcs[cur_head][m];
        if(index_in_valids >= 0){
          ms_left.push_back(m);
          left_variables.push_back(variables[index_in_valids]);
        }
      }
      // build
      AD3::FactorHead *factor = new AD3::FactorHead;
      factor->Initialize(cur_head, false, gs, ms_left, 
        po2sib, po2sib_left_idxes[cur_head], po2g, po2g_left_idxes[cur_head], po3gsib, po3gsib_left_idxes[cur_head]);
      vector<double> additional_log_potentials = po2sib_left_scores[cur_head];
      additional_log_potentials.insert(additional_log_potentials.end(), po2g_left_scores[cur_head].begin(), po2g_left_scores[cur_head].end());
      additional_log_potentials.insert(additional_log_potentials.end(), po3gsib_left_scores[cur_head].begin(), po3gsib_left_scores[cur_head].end());
      factor->SetAdditionalLogPotentials(additional_log_potentials);
      factor_graph->DeclareFactor(factor, left_variables, true);
    }
    // collect ms-right and build
    if(true){
      // collect
      vector<int> ms_right;
      vector<AD3::BinaryVariable*> right_variables = gs_variables;
      for(int m = cur_head + 1; m < slen; m++){
        int index_in_valids = index_arcs[cur_head][m];
        if(index_in_valids >= 0){
          ms_right.push_back(m);
          right_variables.push_back(variables[index_in_valids]);
        }
      }
      // build
      AD3::FactorHead *factor = new AD3::FactorHead;
      factor->Initialize(cur_head, true, gs, ms_right,
        po2sib, po2sib_right_idxes[cur_head], po2g, po2g_right_idxes[cur_head], po3gsib, po3gsib_right_idxes[cur_head]);
      vector<double> additional_log_potentials = po2sib_right_scores[cur_head];
      additional_log_potentials.insert(additional_log_potentials.end(), po2g_right_scores[cur_head].begin(), po2g_right_scores[cur_head].end());
      additional_log_potentials.insert(additional_log_potentials.end(), po3gsib_right_scores[cur_head].begin(), po3gsib_right_scores[cur_head].end());
      factor->SetAdditionalLogPotentials(additional_log_potentials);
      factor_graph->DeclareFactor(factor, right_variables, true);
    }
  }

#ifdef  MY_DEBUG
  /*// debug print info
  (static_cast<AD3::FactorTree*>(factor_graph->GetFactor(0)))->info(cerr, nullptr, nullptr);
  for(int i = 1; i < factor_graph->GetNumFactors(); i++){
    (static_cast<AD3::FactorHead*>(factor_graph->GetFactor(i)))->info(cerr, nullptr, nullptr);
  }*/
  // verboisty
  factor_graph->SetVerbosity(3);
#endif //  MY_DEBUG

  // =====
  // Run AD3
  vector<double> posteriors;
  vector<double> additional_posteriors;
  double value_ref;
  double *value = &value_ref;
  //
#ifdef USE_PSDD
  factor_graph->SetEtaPSDD(1.0);
  factor_graph->SetMaxIterationsPSDD(500);
  factor_graph->SolveLPMAPWithPSDD(&posteriors, &additional_posteriors, value);
#else
  factor_graph->SetMaxIterationsAD3(500);
  //factor_graph->SetMaxIterationsAD3(200);
  factor_graph->SetEtaAD3(0.05);
  factor_graph->AdaptEtaAD3(true);
  factor_graph->SetResidualThresholdAD3(1e-3);
  //factor_graph->SetResidualThresholdAD3(1e-6);
  factor_graph->SolveLPMAPWithAD3(&posteriors, &additional_posteriors, value);
#endif

#ifdef  MY_DEBUG
  /*// debug print info
  (static_cast<AD3::FactorTree*>(factor_graph->GetFactor(0)))->info(cerr, nullptr, nullptr);
  for(int i = 1; i < factor_graph->GetNumFactors(); i++){
    (static_cast<AD3::FactorHead*>(factor_graph->GetFactor(i)))->info(cerr, nullptr, nullptr);
  }*/
#endif //  MY_DEBUG

  delete factor_graph;
  // =====
  // final decode with posteriors
  mst_decode(slen, projective, so1_hs, so1_ms, posteriors, output_heads, output_value);
  return;
}

// interface
#ifndef MY_DEBUG
template<typename T>
vector<T> array2vector_1d(py::array_t<T>* arr){
  py::buffer_info buf = arr->request();
  CHECK(buf.ndim == 1, "Err: not equal dim == 1");
  int size = buf.size;
  T* ptr1 = static_cast<T*>(buf.ptr);
  // copy here. todo(+2): any way to tmply borrow the data?
  vector<T> ret(ptr1, ptr1 + size);
  return ret;
}

vector<int> parse2(const int slen, const int projective,
  const int use_o1, py::array_t<int>* o1_masks, py::array_t<double>* o1_scores,
  const int use_o2sib, py::array_t<int>* o2sib_ms, py::array_t<int>* o2sib_hs, py::array_t<int>* o2sib_ss, py::array_t<double>* o2sib_scores,
  const int use_o2g, py::array_t<int>* o2g_ms, py::array_t<int>* o2g_hs, py::array_t<int>* o2g_gs, py::array_t<double>* o2g_scores,
  const int use_o3gsib, py::array_t<int>* o3gsib_ms, py::array_t<int>* o3gsib_hs, py::array_t<int>* o3gsib_ss, py::array_t<int>* o3gsib_gs, py::array_t<double>* o3gsib_scores)
{
  O1Pack* ppo1 = nullptr;
  O2SibPack* ppo2sib = nullptr;
  O2gPack* ppo2g = nullptr;
  O3gsibPack* ppo3gsib = nullptr;
  // o1
  CHECK(use_o1, "Error: no po1 info provided!");
  vector<int> po1_masks = array2vector_1d(o1_masks);
  vector<double> po1_scores = array2vector_1d(o1_scores);
  // todo(note): exclude [0,0], treat (g,h) specially
  po1_masks[0] = 0;
  //
  ppo1 = new O1Pack(po1_masks, po1_scores);
  // other vectors
  vector<int> po2sib_hs, po2sib_ms, po2sib_ss, po2g_hs, po2g_ms, po2g_gs, po3gsib_hs, po3gsib_ms, po3gsib_ss, po3gsib_gs;
  vector<double> po2sib_scores, po2g_scores, po3gsib_scores;
  // o2sib
  if(use_o2sib){
    po2sib_ms = array2vector_1d(o2sib_ms);
    po2sib_hs = array2vector_1d(o2sib_hs);
    po2sib_ss = array2vector_1d(o2sib_ss);
    po2sib_scores = array2vector_1d(o2sib_scores);
    ppo2sib = new O2SibPack(po2sib_hs, po2sib_ms, po2sib_ss, po2sib_scores);
  }
  // o2g
  if(use_o2g){
    po2g_ms = array2vector_1d(o2g_ms);
    po2g_hs = array2vector_1d(o2g_hs);
    po2g_gs = array2vector_1d(o2g_gs);
    po2g_scores = array2vector_1d(o2g_scores);
    ppo2g = new O2gPack(po2g_hs, po2g_ms, po2g_gs, po2g_scores);
  }
  // o3gsib
  if(use_o3gsib){
    po3gsib_ms = array2vector_1d(o3gsib_ms);
    po3gsib_hs = array2vector_1d(o3gsib_hs);
    po3gsib_ss = array2vector_1d(o3gsib_ss);
    po3gsib_gs = array2vector_1d(o3gsib_gs);
    po3gsib_scores = array2vector_1d(o3gsib_scores);
    ppo3gsib = new O3gsibPack(po3gsib_hs, po3gsib_ms, po3gsib_ss, po3gsib_gs, po3gsib_scores);
  }
  // decode
  vector<int> ret(slen, -1);
  double v;
  parse_with_packs(slen, projective!=0, ppo1, ppo2sib, ppo2g, ppo3gsib, &ret, &v);
  // delete
  delete ppo1;
  delete ppo2sib;
  delete ppo2g;
  delete ppo3gsib;
  return ret;
}

//c++ -O3 -Wall -Wno-sign-compare -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I./ad3/ -I./Eigen -I. ./ad3/*.cpp algo.cpp parser2.cpp -o parser2`python3-config --extension-suffix`
// default policy is return_value_policy::take_ownership when returning pointer
PYBIND11_MODULE(parser2, m) {
  m.doc() = "The high-order graph parsing algorithm with AD3"; // optional module docstring
  m.def("parse2", &parse2, "A function for high-order parsing");
}
#endif

// for debug
/*
b parser2.cpp:300
p slen
p *po1_masks._M_impl._M_start@25
p po2sib_hs.size()
p po2g_hs.size()
p po3gsib_hs.size()
b parser2.cpp:62
*/

//#define MY_DEBUG
// c++ -g -DMY_DEBUG -Wall -Wno-sign-compare -std=c++11 -I./ad3/ -I./Eigen -I. ./ad3/*.cpp algo.cpp parser2.cpp
#ifdef MY_DEBUG
template<typename T>
vector<T> read_vector_1d(std::istream& fin){
  // first read size
  int size = 0;
  fin >> size;
  vector<T> ret(size);
  for(int i = 0; i < size; i++){
    fin >> ret[i];
  }
  return ret;
}

void my_debug(const char* data_fname){
  auto fin = std::ifstream(data_fname);
  int count = 0;
  while(fin){
    // read one record
    // start
    int slen, projective;
    int use_o1, use_o2sib, use_o2g, use_o3gsib;
    fin >> slen >> projective >> use_o1 >> use_o2sib >> use_o2g >> use_o3gsib;
    if(!fin)
      break;
    // o1
    auto po1_masks = read_vector_1d<int>(fin);
    auto po1_scores = read_vector_1d<double>(fin);
    po1_masks[0] = 0;
    O1Pack* ppo1 = use_o1 ? (new O1Pack(po1_masks, po1_scores)) : nullptr;
    // o2sib
    auto po2sib_ms = read_vector_1d<int>(fin);
    auto po2sib_hs = read_vector_1d<int>(fin);
    auto po2sib_ss = read_vector_1d<int>(fin);
    auto po2sib_scores = read_vector_1d<double>(fin);
    O2SibPack* ppo2sib = use_o2sib ? (new O2SibPack(po2sib_hs, po2sib_ms, po2sib_ss, po2sib_scores)) : nullptr;
    // o2g
    auto po2g_ms = read_vector_1d<int>(fin);
    auto po2g_hs = read_vector_1d<int>(fin);
    auto po2g_gs = read_vector_1d<int>(fin);
    auto po2g_scores = read_vector_1d<double>(fin);
    O2gPack* ppo2g = use_o2g ? (new O2gPack(po2g_hs, po2g_ms, po2g_gs, po2g_scores)) : nullptr;
    // o3gsib
    auto po3gsib_ms = read_vector_1d<int>(fin);
    auto po3gsib_hs = read_vector_1d<int>(fin);
    auto po3gsib_ss = read_vector_1d<int>(fin);
    auto po3gsib_gs = read_vector_1d<int>(fin);
    auto po3gsib_scores = read_vector_1d<double>(fin);
    O3gsibPack* ppo3gsib = use_o3gsib ? (new O3gsibPack(po3gsib_hs, po3gsib_ms, po3gsib_ss, po3gsib_gs, po3gsib_scores)) : nullptr;
    //
    // decode
    vector<int> ret(slen, -1);
    double v;
    parse_with_packs(slen, projective!=0, ppo1, ppo2sib, ppo2g, ppo3gsib, &ret, &v);
    // delete
    delete ppo1;
    delete ppo2sib;
    delete ppo2g;
    delete ppo3gsib;
    count++;
  }
}

int main(int argc, char** argv){
  //my_debug(argv[1]);
  my_debug("data.txt");
}

#endif // MY_DEBUG
