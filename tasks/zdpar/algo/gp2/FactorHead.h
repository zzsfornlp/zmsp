//

// Factor for one head (supporting features up to o3gsib)

#ifndef FACTOR_HEAD_H
#define FACTOR_HEAD_H

#include <algorithm>
#include "ad3/GenericFactor.h"
#include "utils.h"

using std::ostream;

// todo(note): no considering [0,0] other than the special [g,h]
namespace AD3 {
  class FactorHead: public GenericFactor{
  public:
    FactorHead() {}
    virtual ~FactorHead() { ClearActiveSet(); }

    // Print as a string.
    void Print(ostream& stream) {
      stream << "HEAD_AUTOMATON up to o3gsib (details omitted)";
      Factor::Print(stream);
    }

    void info(ostream& stream, const vector<double>* variable_scores, const vector<double>* additional_scores){
      vector<double> self_variable_scores;
      vector<double> self_additional_scores;
      // get self if nullptr
      if(variable_scores == nullptr){
        for(auto* v : binary_variables_){
          self_variable_scores.push_back(v->GetLogPotential());
        }
        variable_scores = &self_variable_scores;
      }
      if(additional_scores == nullptr){
        additional_scores = &additional_log_potentials_;
      }
      // collect and print info
      // first
      stream << "FactorHead: head=" << cur_head_ << ", is_right=" << is_right_ << endl;
      // gs
      if(head0_){
        CHECK(cur_head_==0, "");
        CHECK(length_gs_ == 1, "");
        CHECK(base_gs_ == 0, "");
        stream << "gs: With only Dummy 0" << endl;
      }
      else{
        stream << "gs: ";
        for(int i = 0; i < rindex_incoming_.size(); i++){
          stream << rindex_incoming_[i] << "(" << variable_scores->at(i) << ")" << " ";
        }
        stream << endl;
        CHECK(base_gs_ == rindex_incoming_.size(), "");
      }
      // ms
      stream << "ms: ";
      for(int i = 0; i < rindex_modifiers_.size(); i++){
        stream << rindex_modifiers_[i] << "(" << variable_scores->at(i+base_gs_) << ")" << " ";
      }
      stream << endl;
      CHECK(base_gs_ + length_ms_ == variable_scores->size(), "");
      // addional
      int cur_idx_base = 0;
      vector<int> cur_ms, cur_hs, cur_ss, cur_gs;
      vector<double> cur_scores;
      // o2sib
      if(has_o2sib_){
        for(int m = 0; m < length_ms_; m++){
          for(int s = 0; s < length_ms_; s++){
            int cur_idx = index_siblings_[m][s];
            if(cur_idx >= 0){
              cur_ms.push_back(rindex_modifiers_[m]); cur_hs.push_back(cur_head_); cur_ss.push_back(rindex_modifiers_[s]);
              cur_scores.push_back(additional_scores->at(cur_idx_base));
              cur_idx_base++;
            }
          }
        }
        debug_print(stream, &cur_ms, &cur_hs, &cur_ss, nullptr, &cur_scores);
        cur_ms.clear(); cur_hs.clear(); cur_ss.clear(); cur_scores.clear();
      }
      // o2g
      if(has_o2g_){
        for(int m = 0; m < length_ms_; m++){
          for(int g = 0; g < length_gs_; g++){
            int cur_idx = index_grandparents_[g][m];
            if(cur_idx >= 0){
              cur_ms.push_back(rindex_modifiers_[m]); cur_hs.push_back(cur_head_); cur_gs.push_back(rindex_incoming_[g]);
              cur_scores.push_back(additional_scores->at(cur_idx_base));
              cur_idx_base++;
            }
          }
        }
        debug_print(stream, &cur_ms, &cur_hs, nullptr, &cur_gs, &cur_scores);
        cur_ms.clear(); cur_hs.clear(); cur_gs.clear(); cur_scores.clear();
      }
      // o3gsib
      if(has_o3gsib_){
        for(int m = 0; m < length_ms_; m++){
          for(int s = 0; s < length_ms_; s++){
            for(int g = 0; g < length_gs_; g++){
              int cur_idx = index_grandsiblings_[g][m][s];
              if(cur_idx >= 0){
                cur_ms.push_back(rindex_modifiers_[m]); cur_hs.push_back(cur_head_); cur_ss.push_back(rindex_modifiers_[s]); cur_gs.push_back(rindex_incoming_[g]);
                cur_scores.push_back(additional_scores->at(cur_idx_base));
                cur_idx_base++;
              }
            }
          }
        }
        debug_print(stream, &cur_ms, &cur_hs, &cur_ss, &cur_gs, &cur_scores);
      }
      //
      CHECK(cur_idx_base == additional_scores->size(), "");
    }

    // =====
    // main ones
  private:
    // get
    inline double get_o2sib_score(const vector<double> &additional_log_potentials, int m, int s){
      if(has_o2sib_){
        int idx = index_siblings_[m][s];
        CHECK(idx >= 0, "Illegal index");
        return additional_log_potentials[idx];
      }
      else return 0.;
    }

    inline double get_o2g_score(const vector<double> &additional_log_potentials, int g, int m){
      if(has_o2g_){
        int idx = index_grandparents_[g][m];
        CHECK(idx >= 0, "Illegal index");
        return additional_log_potentials[idx];
      }
      else return 0.;
    }

    inline double get_o3gsib_score(const vector<double> &additional_log_potentials, int g, int m, int s){
      if(has_o3gsib_){
        int idx = index_grandsiblings_[g][m][s];
        CHECK(idx >= 0, "Illegal index");
        return additional_log_potentials[idx];
      }
      else return 0.;
    }

    // set
    inline void add_o2sib_weight(vector<double> *additional_posteriors, double weight, int m, int s){
      if(has_o2sib_){
        int idx = index_siblings_[m][s];
        CHECK(idx >= 0, "Illegal index");
        (*additional_posteriors)[idx] += weight;
      }
    }

    inline void add_o2g_weight(vector<double> *additional_posteriors, double weight, int g, int m){
      if(has_o2g_){
        int idx = index_grandparents_[g][m];
        CHECK(idx >= 0, "Illegal index");
        (*additional_posteriors)[idx] += weight;
      }
    }

    inline void add_o3gsib_weight(vector<double> *additional_posteriors, double weight, int g, int m, int s){
      if(has_o3gsib_){
        int idx = index_grandsiblings_[g][m][s];
        CHECK(idx >= 0, "Illegal index");
        (*additional_posteriors)[idx] += weight;
      }
    }

  public:
    // Get the combinatory best solution
    // variable_log_potentials: (g,h)+(h,m), additional_log_potentials: o2sib+o2g+o3gsib
    // -- todo(note): and be careful about [0,0]
    void Maximize(const vector<double> &variable_log_potentials,
      const vector<double> &additional_log_potentials,
      Configuration &configuration,
      double *value) {
      // Decode maximizing over the grandparents and using the Viterbi algorithm as an inner loop.
      int num_g = length_gs_;
      int num_m = length_ms_;
      // -----
      // overall best
      int overall_best_grandparent = -1;
      vector<int> overall_best_modifiers;
      double overall_best_score = MY_NEGINF;
      // Run Viterbi for each possible grandparent.
      for(int g = 0; g < num_g; ++g) {
        int orig_g = rindex_incoming_[g];
        // gh score should be added at the final step!
        double gh_score = (!head0_) ? variable_log_potentials[g] : 0.;  // no gh score or simply ignore (0,0)
        // 0 as the start, thus the recording idxes are all +1
        // these means up-to-this-step(and selecting current, what is the best history)
        vector<int> best_path(num_m + 1, 0);  // for each step, the best prev-m (in recording-m)
        vector<double> best_scores(num_m + 1, MY_NEGINF);  // best scores for each step
        // starting best is selecting nothing
        int best_ending = 0;
        double best_score = 0.;
        best_scores[0] = 0.;
        // for each mod as step
        // todo(note): not-optimized for no-sib situation
        for(int m = 0; m < num_m; ++m) {
          int orig_m = rindex_modifiers_[m];
          if(orig_g == orig_m)
            continue;  // g cannot be m!
          double o2g_score = get_o2g_score(additional_log_potentials, g, m);
          double hm_score = variable_log_potentials[m+base_gs_];  // offset the gh ones
          int recording_m = m + 1;  // step number, since padding a 0 as start state
          // adding current m, find best prev recording-m
          // first the special one for the starting mod: [m, m] means this
          int cur_best_prev_rm = 0;
          double cur_best_score = get_o2sib_score(additional_log_potentials, m, m) + get_o3gsib_score(additional_log_potentials, g, m, m);
          //
          // for other nonzero ones
          for(int s = 0; s < m; s++){
            int orig_s = rindex_modifiers_[s];
            if(orig_g == orig_s)
              continue;  // g cannot be s!
            int mr = s + 1;
            double cur_one_score = best_scores[mr];
            cur_one_score += get_o2sib_score(additional_log_potentials, m, s) + get_o3gsib_score(additional_log_potentials, g, m, s);
            if(cur_one_score > cur_best_score){
              cur_best_score = cur_one_score;
              cur_best_prev_rm = mr;
            }
          }
          // update for one g
          cur_best_score += o2g_score + hm_score;
          best_path[recording_m] = cur_best_prev_rm;
          best_scores[recording_m] = cur_best_score;
          if(cur_best_score > best_score){
            best_score = cur_best_score;
            best_ending = recording_m;
          }
        }
        // update overall
        double best_full_score = best_score + gh_score;
        if(best_full_score > overall_best_score){
          overall_best_score = best_full_score;
          overall_best_grandparent = g;
          // back tracking modifiers
          overall_best_modifiers.clear();
          for(int cur_mr = best_ending; cur_mr > 0; cur_mr=best_path[cur_mr]){
            overall_best_modifiers.push_back(cur_mr-1);  // remember to get real idx
          }
          std::reverse(overall_best_modifiers.begin(), overall_best_modifiers.end());  // correct order
        }
      }
      // Now write the configuration.
      vector<int> *grandparent_modifiers = static_cast<vector<int>*>(configuration);
      grandparent_modifiers->push_back(overall_best_grandparent);
      grandparent_modifiers->insert(grandparent_modifiers->end(), overall_best_modifiers.begin(), overall_best_modifiers.end());
      *value = overall_best_score;
    }

    // Compute the score of a given assignment.
    void Evaluate(const vector<double> &variable_log_potentials,
      const vector<double> &additional_log_potentials,
      const Configuration configuration,
      double *value) {
      const vector<int>* grandparent_modifiers = static_cast<const vector<int>*>(configuration);
      double tmp_value = 0.;
      int g = (*grandparent_modifiers)[0];
      if(!head0_)
        tmp_value += variable_log_potentials[g];
      int prev = -1;
      for(int i = 1; i < grandparent_modifiers->size(); ++i) {
        int m = (*grandparent_modifiers)[i];
        int s = (prev < 0) ? m : prev;
        tmp_value += variable_log_potentials[base_gs_+m];
        // the three additional ones
        tmp_value += get_o2sib_score(additional_log_potentials, m, s);
        tmp_value += get_o2g_score(additional_log_potentials, g, m);
        tmp_value += get_o3gsib_score(additional_log_potentials, g, m, s);
        prev = m;
      }
      *value = tmp_value;
    }

    // Given a configuration with a probability (weight),
    // increment the vectors of variable and additional posteriors.
    void UpdateMarginalsFromConfiguration(
      const Configuration &configuration,
      double weight,
      vector<double> *variable_posteriors,
      vector<double> *additional_posteriors) {
      const vector<int>* grandparent_modifiers = static_cast<const vector<int>*>(configuration);
      int g = (*grandparent_modifiers)[0];
      if(!head0_)
        (*variable_posteriors)[g] += weight;
      int prev = -1;
      for(int i = 1; i < grandparent_modifiers->size(); ++i) {
        int m = (*grandparent_modifiers)[i];
        int s = (prev < 0) ? m : prev;
        (*variable_posteriors)[base_gs_+m] += weight;
        // the three additional ones
        add_o2sib_weight(additional_posteriors, weight, m, s);
        add_o2g_weight(additional_posteriors, weight, g, m);
        add_o3gsib_weight(additional_posteriors, weight, g, m, s);
        prev = m;
      }
    }

    // =====
    // configuration

    // Count how many common values two configurations have.
    int CountCommonValues(const Configuration &configuration1, const Configuration &configuration2) {
      const vector<int> *values1 = static_cast<const vector<int>*>(configuration1);
      const vector<int> *values2 = static_cast<const vector<int>*>(configuration2);
      int count = 0;
      if((*values1)[0] == (*values2)[0]) ++count; // Grandparents matched.
      int j = 1;
      for(int i = 1; i < values1->size(); ++i) {
        for(; j < values2->size(); ++j) {
          if((*values2)[j] >= (*values1)[i]) break;
        }
        if(j < values2->size() && (*values2)[j] == (*values1)[i]) {
          ++count;
          ++j;
        }
      }
      return count;
    }

    // Check if two configurations are the same.
    bool SameConfiguration(
      const Configuration &configuration1,
      const Configuration &configuration2) {
      const vector<int> *values1 = static_cast<const vector<int>*>(configuration1);
      const vector<int> *values2 = static_cast<const vector<int>*>(configuration2);
      if(values1->size() != values2->size()) return false;
      for(int i = 0; i < values1->size(); ++i) {
        if((*values1)[i] != (*values2)[i]) return false;
      }
      return true;
    }

    // Delete configuration.
    void DeleteConfiguration(
      Configuration configuration) {
      vector<int> *values = static_cast<vector<int>*>(configuration);
      delete values;
    }

    Configuration CreateConfiguration() {
      // The first element is the index of the grandparent.
      // The remaining elements are the indices of the modifiers.
      vector<int>* grandparent_modifiers = new vector<int>;
      return static_cast<Configuration>(grandparent_modifiers);
    }

  public:
  // =====
  // init

  // h, gs, ms: original idxes (in sentence) for the possible gs and ms
  // -- gs is l2r, ms is l2r/r2l according to the direction
  // the rest arguments indicate the high-order parts, idxes are the ones that are effective
  // gs can include 0, but ms does not has 0!
  // todo(note): no checking here, assuming at the outside the processes are correct without including strange ones
  void Initialize(const int h, const bool is_right, const vector<int> &gs, const vector<int> &ms,
    const O2SibPack* po2sib, const vector<int>& po2sib_idxes,
    const O2gPack* po2g, const vector<int>& po2g_idxes, 
    const O3gsibPack* po3gsib, const vector<int>& po3gsib_idxes) {
    // =====
    cur_head_ = h;
    is_right_ = is_right;
    if(h == 0){
      head0_ = true;
      CHECK(is_right, "Err: 0-left");
      CHECK(gs.size() == 0, "Err: head-of-0");
    }
    else{
      head0_ = false;
    }
    // length/idxes are relative to the head position.
    length_gs_ = head0_ ? 1 : gs.size();
    length_ms_ = ms.size();  // no stopping (outmost) part!
    base_gs_ = gs.size();
    has_o2sib_ = has_o2g_ = has_o3gsib_ = false;
    if(po2sib){
      index_siblings_.assign(length_ms_, vector<int>(length_ms_, -1));
      has_o2sib_ = true;
    }
    if(po2g){
      index_grandparents_.assign(length_gs_, vector<int>(length_ms_, -1));
      has_o2g_ = true;
    }
    if(po3gsib){
      index_grandsiblings_.assign(length_gs_, vector<vector<int> >(length_ms_, vector<int>(length_ms_, -1)));
      has_o3gsib_ = true;
    }
    need_sib_ = (has_o2sib_ || has_o3gsib_);
    need_g_ = (has_o2g_ || has_o3gsib_);
    // =====
    // Create a temporary index of modifiers.
    // todo(note): here, no special idx0, since o2sib[m][m] means start-mod; expect for [0][m] for root-self-loop
    index_modifiers_.clear();
    for(int k = 0; k < ms.size(); ++k) {
      int m = ms[k];
      int position = is_right ? (m - h) : (h - m);
      index_modifiers_.resize(position + 1, -1);
      index_modifiers_[position] = k;
    }
    // Create a temporary index of grandparents.
    // todo(note): special treating for (0,0)
    index_incoming_.clear();
    if(cur_head_ == 0){
      // specially put 0 here
      index_incoming_.push_back(0);
    }
    else{
      for(int k = 0; k < gs.size(); ++k) {
        int g = gs[k];
        int position = g;
        index_incoming_.resize(position + 1, -1);
        index_incoming_[position] = k;
      }
    }
    // =====
    int cur_idx_base = 0;
    // Construct index of siblings.
    for(int r = 0; r < po2sib_idxes.size(); r++){
      int ri = po2sib_idxes[r];
      //int cur_h = po2sib->idxes_h[ri];
      int m = po2sib->idxes_m[ri];
      int s = po2sib->idxes_s[ri];
      // change idx
      int position_modifier = is_right ? (m - h) : (h - m);
      int position_sibling = is_right ? (s - h) : (h - s);
      int index_modifier = index_modifiers_[position_modifier];
      int index_sibling = index_modifiers_[position_sibling];
      // final overall idx
      index_siblings_[index_modifier][index_sibling] = cur_idx_base;
      cur_idx_base++;
    }
    // Construct index of grandparents.
    for(int r = 0; r < po2g_idxes.size(); r++){
      int ri = po2g_idxes[r];
      //int cur_h = po2g->idxes_h[ri];
      int m = po2g->idxes_m[ri];
      int g = po2g->idxes_g[ri];
      // change idx
      int position_modifier = is_right ? (m - h) : (h - m);
      int position_grandparent = g;
      int index_modifier = index_modifiers_[position_modifier];
      int index_grandparent = index_incoming_[position_grandparent];
      // final idx
      index_grandparents_[index_grandparent][index_modifier] = cur_idx_base;
      cur_idx_base++;
    }
    // Construct index of grandsiblings.
    for(int r = 0; r < po3gsib_idxes.size(); r++){
      int ri = po3gsib_idxes[r];
      //int cur_h = po3gsib->idxes_h[ri];
      int m = po3gsib->idxes_m[ri];
      int s = po3gsib->idxes_s[ri];
      int g = po3gsib->idxes_g[ri];
      // change idx
      int position_modifier = is_right ? (m - h) : (h - m);
      int position_sibling = is_right ? (s - h) : (h - s);
      int position_grandparent = g;
      int index_modifier = index_modifiers_[position_modifier];
      int index_sibling = index_modifiers_[position_sibling];
      int index_grandparent = index_incoming_[position_grandparent];
      // final overall idx
      index_grandsiblings_[index_grandparent][index_modifier][index_sibling] = cur_idx_base;
      cur_idx_base++;
    }
    // reverse index
    rindex_incoming_.assign(length_gs_, -1);
    rindex_modifiers_.assign(length_ms_, -1);
    for(unsigned i = 0; i < index_incoming_.size(); i++){
      int x = index_incoming_[i];
      if(x >= 0) rindex_incoming_[x] = i;
    }
    for(unsigned i = 0; i < index_modifiers_.size(); i++){
      int x = index_modifiers_[i];
      if(x >= 0) rindex_modifiers_[x] = ((is_right) ? (h+i) : (h-i));
    }
  }

  private:
  //
  int cur_head_;
  bool head0_;  // cur_head is 0
  bool is_right_;
  // overall (valid) lengths
  int length_gs_;
  int length_ms_;
  int base_gs_;  // basis as the varaible-offset (excluding 0,0)
  // valid pieces
  bool has_o2sib_;
  bool has_o2g_;
  bool has_o3gsib_;
  bool need_sib_;
  bool need_g_;
  // idxes from relative-idx to flattened-additional-scores
  vector<vector<int> > index_siblings_;  // [len_ms_m, len_ms_s]
  vector<vector<int> > index_grandparents_;  // [len_gs, len_ms]
  vector<vector<vector<int> > > index_grandsiblings_;  // // [len_gs, len_ms_m, len_ms_s]
  // for debugging
  // orig_idx -> cur_idx
  vector<int> index_incoming_;
  vector<int> index_modifiers_;
  // reverse: cur_idx -> orig_idx/orig_distance
  vector<int> rindex_incoming_;
  vector<int> rindex_modifiers_;
};
}

#endif // !FACTOR_HEAD_H
