//

// Factor for the overall tree structure

#ifndef FACTOR_TREE_H
#define FACTOR_TREE_H

#include "utils.h"

// whether put edge's o1-score as factor-variable (which will be divided to all the links in the algorithm) or additional-score of this TreeFactor
// #define FACTOR_TREE_ADDITIONAL_SCORE

namespace AD3 {
  class FactorTree: public GenericFactor {
  public:
    FactorTree() {}
    virtual ~FactorTree() { ClearActiveSet(); }

    // Print as a string.
    void Print(ostream& stream) {
      stream << "ARBORESCENCE (details omitted)";
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
      // check
      CHECK(variable_scores->size() == hs_->size(), "Err: size mismatch");
#ifdef FACTOR_TREE_ADDITIONAL_SCORE
      CHECK(additional_scores->size() == variable_scores->size(), "Err: size mismatch");
#else
      CHECK(additional_scores->size() == 0, "Err: size mismatch");
#endif
      // print
      stream << "FactorTree: slen=" << length_ << ", narc=" << variable_scores->size() << ", projective=" << projective_ << endl;
      int count = 0;
      vector<int> cur_ms;
      vector<int> cur_hs;
      vector<double> cur_scores;
      for(int m = 0; m < length_; m++){
        for(int h = 0; h < length_; h++){
          int cur_i = index_arcs_[h][m];
          if(cur_i >= 0){
            cur_ms.push_back(m);
            cur_hs.push_back(h);
            cur_scores.push_back(variable_scores->at(cur_i));
          }
        }
      }
      CHECK(variable_scores->size() == cur_ms.size(), "Err: size mismatch");
      // print2
      debug_print(stream, &cur_ms, &cur_hs, nullptr, nullptr, &cur_scores);
    }

    // =====
    // main ones
    // use additional_log_potentials as extra o1 scores

    // maximize using specific algorithms
    void Maximize(const vector<double> &variable_log_potentials,
      const vector<double> &additional_log_potentials,
      Configuration &configuration,
      double *value) {
      vector<int>* heads = static_cast<vector<int>*>(configuration);
#ifdef FACTOR_TREE_ADDITIONAL_SCORE
      vector<double> cur_scores = variable_log_potentials;
      for(int i = 0; i < cur_scores.size(); i++)
        cur_scores[i] += additional_log_potentials[i];
      mst_decode(length_, projective_, *hs_, *ms_, cur_scores, heads, value);
#else
      mst_decode(length_, projective_, *hs_, *ms_, variable_log_potentials, heads, value);
#endif
    }

    // Compute the score of a given assignment.
    void Evaluate(const vector<double> &variable_log_potentials,
      const vector<double> &additional_log_potentials,
      const Configuration configuration,
      double *value) {
      const vector<int> *heads = static_cast<const vector<int>*>(configuration);
      // Heads belong to {0,1,2,...}
      *value = 0.0;
      for(int m = 1; m < heads->size(); ++m) {
        int h = (*heads)[m];
        int index = index_arcs_[h][m];
        *value += variable_log_potentials[index];
#ifdef FACTOR_TREE_ADDITIONAL_SCORE
        *value += additional_log_potentials[index];
#endif
      }
    }

    // Given a configuration with a probability (weight),
    // increment the vectors of variable and additional posteriors.
    void UpdateMarginalsFromConfiguration(
      const Configuration &configuration,
      double weight,
      vector<double> *variable_posteriors,
      vector<double> *additional_posteriors) {
      const vector<int> *heads = static_cast<const vector<int>*>(configuration);
      for(int m = 1; m < heads->size(); ++m) {
        int h = (*heads)[m];
        int index = index_arcs_[h][m];
        (*variable_posteriors)[index] += weight;
#ifdef FACTOR_TREE_ADDITIONAL_SCORE
        (*additional_posteriors)[index] += weight;
#endif
      }
    }

    // Count how many common values two configurations have.
    int CountCommonValues(const Configuration &configuration1,
      const Configuration &configuration2) {
      const vector<int> *heads1 = static_cast<const vector<int>*>(configuration1);
      const vector<int> *heads2 = static_cast<const vector<int>*>(configuration2);
      int count = 0;
      for(int i = 1; i < heads1->size(); ++i) {
        if((*heads1)[i] == (*heads2)[i]) {
          ++count;
        }
      }
      return count;
    }

    // Check if two configurations are the same.
    bool SameConfiguration(
      const Configuration &configuration1,
      const Configuration &configuration2) {
      const vector<int> *heads1 = static_cast<const vector<int>*>(configuration1);
      const vector<int> *heads2 = static_cast<const vector<int>*>(configuration2);
      for(int i = 1; i < heads1->size(); ++i) {
        if((*heads1)[i] != (*heads2)[i]) return false;
      }
      return true;
    }

    // Delete configuration.
    void DeleteConfiguration(
      Configuration configuration) {
      vector<int> *heads = static_cast<vector<int>*>(configuration);
      delete heads;
    }

    // Create configuration.
    Configuration CreateConfiguration() {
      vector<int>* heads = new vector<int>(length_);
      return static_cast<Configuration>(heads);
    }

  public:
    void Initialize(const bool projective, const int length, const vector<int>* hs, const vector<int>* ms) {
      projective_ = projective;
      length_ = length;
      hs_ = hs;
      ms_ = ms;
      index_arcs_.assign(length, vector<int>(length, -1));
      for(int k = 0; k < hs->size(); ++k) {
        int h = (*hs_)[k];
        int m = (*ms_)[k];
        index_arcs_[h][m] = k;
      }
    }

  private:
    bool projective_; // If true, assume projective trees.
    int length_; // Sentence length (including root symbol).
    const vector<int>* hs_;
    const vector<int>* ms_;
    vector<vector<int> > index_arcs_;
  };
}

#endif // !FACTOR_TREE_H

//TODO: deal with the decoding algo
