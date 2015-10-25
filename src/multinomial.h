#ifndef MULTINOMIAL_H
#define MULTINOMIAL_H

#include <vector>
#include <set>
#include <cassert>
#include <cmath>

#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>



template <typename Count,typename dType>
class multinomial {
  std::vector<int> J;
  std::vector<dType> q;
  boost::random::uniform_int_distribution<Count> unif_int;
  boost::random::uniform_real_distribution<> unif_real;
  std::vector<dType> m_prob, m_logprob;

public:
  multinomial() : unif_real(0.0, 1.0) { }
  multinomial(const std::vector<Count> &counts) : unif_real(0.0, 1.0) { estimate(counts);  }

  void estimate(const std::vector<Count>& counts)
  {
    int k = counts.size();
    Count n = 0;
    m_prob.clear();
    m_prob.resize(k, 0.0);
    m_logprob.clear();
    m_logprob.resize(k, 0.0);
    for (int i=0; i<k; i++)
        n += counts[i];
    for (int i=0; i<k; i++)
    {
        m_prob[i] = static_cast<dType>(counts[i]) / n;
        m_logprob[i] = std::log(m_prob[i]);
    }
    setup(m_prob);
  }

  dType prob(int i) const { return m_prob[i]; }
  dType logprob(int i) const { return m_logprob[i]; }

  template <typename Engine>
  int sample(Engine &eng) const
  {
      int m = unif_int(eng);
      dType p = unif_real(eng);
      int s;
      if (q[m] > p)
	  	  s = m;
      else
        s = J[m];
      assert (s >= 0);
      return s;
  }

private:
 void setup(const std::vector<dType>& probs)
  {
    int k = probs.size();

    unif_int = boost::random::uniform_int_distribution<Count>(0, k-1);
    J.resize(k, -1);
    q.resize(k, 0);
    
    // "small" outcomes (prob < 1/k)
    std::set<int> S;
    std::set<int>::iterator s_it;
    // "large" outcomes (prob >= 1/k)
    std::set<int> L;
    std::set<int>::iterator l_it;
    const dType tol = 1e-3;
    
    for (int i=0; i<k; i++) 
    {
        q[i] = k*probs[i];
        if (q[i] < 1.0)
        {
            S.insert(i);
        }
        else
        {
            L.insert(i);
        } 
    }

    while (S.size() > 0 && L.size() > 0)
    {
        // choose an arbitrary element s from S and l from L
        s_it = S.begin();
        int s = *s_it;
        l_it = L.begin();
        int l = *l_it;

	       // pair up s and (part of) l as its alias
        J[s] = l;
        S.erase(s_it);
        //q[l] = q[l] - (1.0 - q[s]);
	       q[l] = q[l] + q[s] - 1.0; // more stable?

	       // move l from L to S if necessary
        if (q[l] < 1.0)
        {
            S.insert(l);
            L.erase(l_it);
        }
    }

    // any remaining elements must have q/n close to 1, so we leave them alone
    for (s_it = S.begin(); s_it != S.end(); ++s_it) {
      //assert (fabs(q[*s_it] - 1) < tol);
      if (std::fabs(q[*s_it] - 1) > tol)
      {
	       std::cerr << "warning: multinomial: probability differs from one by " << std::fabs(q[*s_it]-1) << std::endl;
      }
      q[*s_it] = 1.0;
    }
    for (l_it = L.begin(); l_it != L.end(); ++l_it) {
      if (std::fabs(q[*l_it] - 1) > tol)
      {
	         std::cerr << "warning: multinomial: probability differs from one by " << std::fabs(q[*l_it]-1) << std::endl;
      }
	  q[*l_it] = 1.0;
    }
  }

};

#endif
