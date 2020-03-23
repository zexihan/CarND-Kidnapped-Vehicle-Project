/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang - functions modifed by C. Haliburton on Dec. 19, 2019
 */

#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  num_particles = 100;                                        // Set the number of particles
  std::default_random_engine gen;                             // Initialize engine for random number creation
  std::normal_distribution<double> dist_x(x, std[0]);         // Create Distribution around initialized 'x'
  std::normal_distribution<double> dist_y(y, std[1]);         // Create Distribution around initialized 'y'
  std::normal_distribution<double> dist_theta(theta, std[2]); // Create Distribution around initialized 'theta'

  for (int p = 0; p < num_particles; ++p)
  {
    Particle particle;                  // create a struct for each particle instantiation
    particle.id = p;                    // each particle is identified
    particle.x = dist_x(gen);           // set particle's x position to a random x based on random Gaussian noise around GPS
    particle.y = dist_y(gen);           // set particle's y position to a random Gaussian noise around GPS
    particle.theta = dist_theta(gen);   // set particle's heading to a random heading based on random Gaussian noise around GPS
    particle.weight = 1.0;              // set particle's weight to 1.0 to start as equal probability
    particles.push_back(particle);      // add particle to particle filter
    weights.push_back(particle.weight); // add particles weight to particle filter weight vector for use in resampling function
  }
  is_initialized = 1; // set particle filter to be initialized to rely on map and measurement data
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  std::default_random_engine gen; // Initialize engine for random number creation needed?
  for (int p = 0; p < num_particles; ++p)
  {                      // For each timestep, predict each particle's next location based on sensor data (velocity and yaw_rate)
    double pred_x = 0.0; // Initialize temp variables
    double pred_y = 0.0;
    double pred_theta = 0.0;
    if (yaw_rate == 0)
    { // avoid divide by zero if yaw_rate is 0.  Calculate straightline motion positions after timestep
      pred_x = particles[p].x + velocity * delta_t * cos(particles[p].theta);
      pred_y = particles[p].y + velocity * delta_t * sin(particles[p].theta);
      pred_theta = particles[p].theta;
    }
    else
    { // if turning, calculate positions after timestep
      pred_x = particles[p].x + velocity / yaw_rate * (sin(particles[p].theta + yaw_rate * delta_t) - sin(particles[p].theta));
      pred_y = particles[p].y + velocity / yaw_rate * (cos(particles[p].theta) - cos(particles[p].theta + yaw_rate * delta_t));
      pred_theta = particles[p].theta + yaw_rate * delta_t;
    }
    std::normal_distribution<double> N_x(pred_x, std_pos[0]); //create normal distribuion around newly predicted particle position.  revisit this as perhaps this should be around the sensor inputs?
    std::normal_distribution<double> N_y(pred_y, std_pos[1]);
    std::normal_distribution<double> N_theta(pred_theta, std_pos[2]);

    particles[p].x = N_x(gen);         //pick a value from the normal distribution for the particles new position x
    particles[p].y = N_y(gen);         //pick a value from the normal distribution for the particles new position y
    particles[p].theta = N_theta(gen); //pick a value from the normal distribution for the particles new position theta
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  for (unsigned int o = 0; o < observations.size(); ++o)
  {                                   //for each of the sensor measurement observations
    double closest_obs_range = 99999; //set the nearest landmark to be far away so that I can iterate through each landmark saving the closest landmark
    int tempID = -1;                  //for error handling ensure the ID is held as a temp non-valid value

    for (unsigned int i = 0; i < predicted.size(); ++i)
    {                                                                                            //iterate through each landmark to find the closest landmark
      double range = dist(observations[o].x, observations[o].y, predicted[i].x, predicted[i].y); // CALCULATE RANGE
      if (range < closest_obs_range)
      {
        closest_obs_range = range; //if the calculated range is less than the last save the id for association
        tempID = predicted[i].id;
      }
    }
    observations[o].id = tempID; //after iterating through each map landmark take the closest and pair with the sensor measurement observation
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  double pWeightTotal = 0.0; //temporary initialization of weight of all particles
  // #1: For each particle, P, with observations in vehicle-grid, translate observations to map-grid (particle data stays as estimate)
  for (int p = 0; p < num_particles; p++)
  {
    LandmarkObs obsTranslated;      //create an instance of LandmarkObs to hold a translated observation
    vector<LandmarkObs> obsTrans_v; //create a vector to hold all translated observations for each particle

    for (unsigned int o = 0; o < observations.size(); ++o)
    { //iterate through each observation to translate it to map coordinates
      obsTranslated.x = particles[p].x + (cos(particles[p].theta) * observations[o].x) - (sin(particles[p].theta) * observations[o].y);
      obsTranslated.y = particles[p].y + (sin(particles[p].theta) * observations[o].x) + (cos(particles[p].theta) * observations[o].y);
      //insert calculation of standard deviation & normal distribution of sensor error?
      obsTranslated.id = o;                //set the id for completeness
      obsTrans_v.push_back(obsTranslated); //assign translated observations to a vector for data handling
    }

    // #2: Eliminate landmarks that are out of range, defined as sensor_range, of particle p
    vector<LandmarkObs> landmarksInRange; //create a vector of landmarks that are in range to speed up dataAssociation
    for (unsigned int l = 0; l < map_landmarks.landmark_list.size(); ++l)
    {
      double landmarkRange = dist(particles[p].x, // Calculate sensor landmark to particle distance
                                  particles[p].y,
                                  map_landmarks.landmark_list[l].x_f,
                                  map_landmarks.landmark_list[l].y_f);
      if (landmarkRange < sensor_range)
      {                                        //check if the landmark is within the sensor range
        landmarksInRange.push_back(LandmarkObs{//if landmark is in range, assign landmark to the vector for passing to nearest neighbour dataAssociation
                                               map_landmarks.landmark_list[l].id_i,
                                               map_landmarks.landmark_list[l].x_f,
                                               map_landmarks.landmark_list[l].y_f});
      }
    }

    // #3: Associate each observation with the closest landmark that is within sensor range as determined in step #2
    dataAssociation(landmarksInRange, obsTrans_v);

    // #4: Calculate the probability for each particle
    double gaussProb = 1.0; //Initialize this instance of ther particle filter's probability placeholder to 1.
    for (unsigned int o = 0; o < obsTrans_v.size(); ++o)
    { //Iterate through each of the observations that has been associated to a landmark
      unsigned int l = 0;
      while (l < landmarksInRange.size())
      { //Iterate through landmarks in range to find nearest neighbour for current observation of current particle
        if (landmarksInRange[l].id == obsTrans_v[o].id)
        {                                           //Find dataAssociate'd nearest neighbour
          gaussProb *= multiv_prob(std_landmark[0], //Calculate multi-variate Gaussian probability and combine with particle's other observations-landmark nearest neighbour probabilities
                                   std_landmark[1],
                                   obsTrans_v[o].x,
                                   obsTrans_v[o].y,
                                   landmarksInRange[l].x,
                                   landmarksInRange[l].y);
          l = landmarksInRange.size(); //terminate while loop assuming that only 1 nearest neighbour is assigned per observation
        }
        else
        {
          l++; //Otherwise iterate through landmarks to find nearest neighbour
        }
      }
    }
    particles[p].weight = gaussProb;     //Assign the multivariate gaussian distribution to the particle's probability
    pWeightTotal += particles[p].weight; //increment the total weight of all particles for use in normalizing
  }

  // #5: After each particle's weight is calculated, normalize to total probability of 1 & populate particle weight vector
  for (int p = 0; p < particles.size(); p++)
  {                                      //increment the total weight of all particles for use in normalizing
    particles[p].weight /= pWeightTotal; //normalize each weight
    weights[p] = particles[p].weight;    //assign to weight vector for use in resampling
  }
}

void ParticleFilter::resample()
{
  std::default_random_engine gen;               //assign to weight vector for use in resampling
  std::discrete_distribution<int> distribution( //create a distribution of all particle probabilities
      weights.begin(),
      weights.end());
  vector<Particle> resampled_particles; //create a prototype instance or resampled particles
  for (int p = 0; p < num_particles; p++)
  { //Iterate through all particles resampling based on the particle probabilities
    resampled_particles.push_back(particles[distribution(gen)]);
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}