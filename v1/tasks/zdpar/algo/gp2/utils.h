// 

#ifndef P2_UTILS_H
#define P2_UTILS_H

// helpful utils

#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>

using std::vector;
using std::string;
using std::cerr;
using std::endl;
using std::ostream;

// =====
// Use four types of sub-trees

// special O1Pack

// Order 1 is different than others
struct O1Pack{
  const vector<int>& masks;  // whether this edge is valid (not pruned) [slen*slen]; should be bool, but make it int for api
  const vector<double>& scores;  // scores for each edge [slen*slen]
  O1Pack(const vector<int>& ma, const vector<double>& ss): masks(ma), scores(ss) {}
};

// The rest uses the indexes

// [h, m, s], when m==s, it means first sib
struct O2SibPack{
  const vector<int>& idxes_h;
  const vector<int>& idxes_m;
  const vector<int>& idxes_s;
  const vector<double>& scores;
  O2SibPack(const vector<int>& h, const vector<int>& m, const vector<int>& s, const vector<double>& ss):
    idxes_h(h), idxes_m(m), idxes_s(s), scores(ss){}
  const unsigned size() const { return scores.size(); }
};

// [h, m, g], when h==0, g==0
struct O2gPack{
  const vector<int>& idxes_h;
  const vector<int>& idxes_m;
  const vector<int>& idxes_g;
  const vector<double>& scores;
  O2gPack(const vector<int>& h, const vector<int>& m, const vector<int>& g, const vector<double>& ss):
    idxes_h(h), idxes_m(m), idxes_g(g), scores(ss){}
  const unsigned size() const { return scores.size(); }
};

// the combined one
struct O3gsibPack{
  const vector<int>& idxes_h;
  const vector<int>& idxes_m;
  const vector<int>& idxes_s;
  const vector<int>& idxes_g;
  const vector<double>& scores;
  O3gsibPack(const vector<int>& h, const vector<int>& m, const vector<int>& s, const vector<int>& g, const vector<double>& ss):
    idxes_h(h), idxes_m(m), idxes_s(s), idxes_g(g), scores(ss){}
  const unsigned size() const { return scores.size(); }
};

// =====
// helpers

const double MY_NEGINF = -1e20;

inline void MY_ERROR(const string& x){
  cerr << "Error: " << x << endl;
  throw x;
}

inline void CHECK(bool v, const string& x){
  if(!v) MY_ERROR(x);
}

template<typename T1, typename T2>
inline void CHECK_GE(T1 x, T2 y){ CHECK(x >= y, "Err: CHECK_GE"); }

template<typename T1, typename T2>
inline void CHECK_LT(T1 x, T2 y){ CHECK(x < y, "Err: CHECK_LT"); }

inline void debug_print(ostream& fout, const vector<int>* ms, const vector<int>* hs, const vector<int>* ss, const vector<int>* gs, const vector<double>* scores){
  int size = scores->size();
  if(ms){
    fout << "m ";
    for(int x : *ms)
      fout << x << " ";
    fout << endl;
  }
  if(hs){
    fout << "h ";
    for(int x : *hs)
      fout << x << " ";
    fout << endl;
  }
  if(ss){
    fout << "s ";
    for(int x : *ss)
      fout << x << " ";
    fout << endl;
  }
  if(gs){
    fout << "g ";
    for(int x : *gs)
      fout << x << " ";
    fout << endl;
  }
  fout << "v ";
  for(double x : *scores)
    fout << x << " ";
  fout << endl;
}

// =====
void mst_decode(int sentence_length, bool projective, const vector<int>& hs, const vector<int>& ms, const vector<double>& scores,
  vector<int> *heads, double *value);

#endif // !P2_UTILS_H
