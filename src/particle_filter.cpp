/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  if (!is_initialized) {
    num_particles = 10;
    vector<double> ones(num_particles, 1.0);
    weights = ones;
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    for (int i = 0; i < num_particles; i++) {
      Particle p{};
      p.x = dist_x(gen);
      p.y = dist_y(gen);
      p.theta = dist_theta(gen);
      p.weight = weights[i];
      particles.push_back(p);
    }
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[i];
    double x = p.x;
    double y = p.y;
    double theta = p.theta;
    double turned_angle = yaw_rate * delta_t;
    x += velocity / yaw_rate * (sin(theta + turned_angle) - sin(theta));
    y += velocity / yaw_rate * (cos(theta) - cos(theta + turned_angle));
    theta += turned_angle;
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std_pos[0]);
    normal_distribution<double> dist_y(y, std_pos[1]);
    normal_distribution<double> dist_theta(theta, std_pos[2]);
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    particles[i] = p;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  for (LandmarkObs &obs : observations) {
    int closest = -1;
    double low_dist = numeric_limits<double>::max();
    for (LandmarkObs pre : predicted) {
      double distance = dist(obs.x, obs.y, pre.x, pre.y);
      if (distance < low_dist) {
        closest = pre.id;
        low_dist = distance;
      }
    }
    obs.id = closest;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  for (Particle &p : particles) {
    vector<Map::single_landmark_s> mlms = map_landmarks.landmark_list;
    vector<LandmarkObs> lmLst;
    for_each(mlms.begin(), mlms.end(), [&lmLst](Map::single_landmark_s slm) { lmLst.push_back({slm.id_i, slm.x_f, slm.y_f}); });
    vector<LandmarkObs> predicted;
    copy_if(lmLst.begin(), lmLst.end(), back_inserter(predicted), [sensor_range, p](LandmarkObs lm) { return dist(p.x, p.y, lm.x, lm.y) <= sensor_range; });
    vector<LandmarkObs> observations_to_p = observations;
    for_each(observations_to_p.begin(), observations_to_p.end(), [p](LandmarkObs &lm) {
        double xo = lm.x;
        double yo = lm.y;
        lm.x = p.x + cos(p.theta) * xo - sin(p.theta) * yo;
        lm.y = p.y + sin(p.theta) * xo + cos(p.theta) * yo;
    });
    dataAssociation(predicted, observations_to_p);
    for (LandmarkObs obs : observations_to_p) {
      Map::single_landmark_s mlm = map_landmarks.landmark_list[obs.id - 1];
      double gauss_norm = 2 * M_PI * std_landmark[0] * std_landmark[1];
      p.weight *= exp(-(pow(obs.x - mlm.x_f, 2) / (2 * pow(std_landmark[0], 2)) + pow(obs.y - mlm.y_f, 2) / (2 * pow(std_landmark[1], 2)))) / gauss_norm;
    }
  }
  double total_weight = 0.0;
  for_each(particles.begin(), particles.end(), [&total_weight](Particle p) { total_weight += p.weight; });
  cout << total_weight << endl;
  for_each(particles.begin(), particles.end(), [total_weight](Particle &p) { p.weight /= total_weight; });
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
