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

#include "particle_filter.h"

std::mt19937 generator;
std::normal_distribution<double> normdist(0.0,1.0);

double rnorm(double mean, double stdd)
{
    return normdist(generator)*stdd+mean;
}

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 200;

	for(int i=0;i!=num_particles;i++)
	{
        Particle newP;
        newP.id = i;
        newP.x = rnorm(x,std[0]);
        newP.y = rnorm(y,std[1]);
        newP.theta = rnorm(theta,std[2]);
        newP.weight = 1;
        particles.push_back(newP);
	}
	weights.resize(num_particles);
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for(int i=0;i!=num_particles;i++)
	{
        if(fabs(yaw_rate) < 0.0001) //yaw rate is too low, use simpler calculation
        {
            particles[i].x += velocity*delta_t*cos(particles[i].theta);
            particles[i].y += velocity*delta_t*sin(particles[i].theta);
        }
        else
        {
            double new_theta = particles[i].theta+yaw_rate*delta_t;
            particles[i].x += velocity/yaw_rate*(sin(new_theta)-sin(particles[i].theta)) + rnorm(0,std_pos[0]);
            particles[i].y += velocity/yaw_rate*(cos(particles[i].theta)-cos(new_theta)) + rnorm(0,std_pos[1]);
            particles[i].theta = new_theta + rnorm(0,std_pos[2]);
        }
        //add noise (important!)
        particles[i].x += rnorm(0,std_pos[0]);
        particles[i].y += rnorm(0,std_pos[1]);
        particles[i].theta += rnorm(0,std_pos[2]);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for(int i=0;i!=num_particles;i++)
	{
        //clear data from previous update
        particles[i].associations.clear();
        particles[i].sense_x.clear();
        particles[i].sense_y.clear();
        //initialize local variables
        double dist;
        double min_dist;
        double min_num;
        double dist_x;
        double dist_y;
        double temp_obs_x;
        double temp_obs_y;
        bool have_correct_meas = false;
        //set weight to 1 - we will multiply it by gaussian values
        particles[i].weight = 1;
        for(int k=0;k!=observations.size();k++) //for all observations...
        {
            min_dist = -1; // no min distance found
            //convert coordinates
            temp_obs_x = observations[k].x*cos(particles[i].theta) - observations[k].y*sin(particles[i].theta) + particles[i].x;
            temp_obs_y = observations[k].x*sin(particles[i].theta) + observations[k].y*cos(particles[i].theta) + particles[i].y;
            for(int j=0;j!=map_landmarks.landmark_list.size();j++)  //go throught all landmarks...
            {
                dist_x = map_landmarks.landmark_list[j].x_f - temp_obs_x;
                dist_y = map_landmarks.landmark_list[j].y_f - temp_obs_y;
                dist = sqrt(dist_x*dist_x + dist_y*dist_y); //get distance
                if(dist <= sensor_range)    //if the landmark is in range, look for the closest
                {
                    if(min_dist == -1)  //first step initializes the distance
                    {
                        min_dist = dist;
                        min_num = j;
                    }
                    else
                    {
                        if(dist < min_dist) //found closer? update
                        {
                            min_dist = dist;
                            min_num = j;
                        }
                    }
                }
            }
            if(min_dist != -1)  //if there is at least one correct measurement...
            {   //push x and y values, association landmark to particle arrays
                particles[i].sense_x.push_back(temp_obs_x);
                particles[i].sense_y.push_back(temp_obs_y);
                particles[i].associations.push_back(map_landmarks.landmark_list[min_num].id_i);
                //calculate multivariate gaussian and update weight
                double xerrexep = map_landmarks.landmark_list[min_num].x_f - temp_obs_x;
                double yerrexep = map_landmarks.landmark_list[min_num].y_f - temp_obs_y;

                particles[i].weight *= exp(-0.5*(xerrexep*xerrexep/std_landmark[0]/std_landmark[0] +
                yerrexep*yerrexep/std_landmark[1]/std_landmark[1]))/(2*M_PI*std_landmark[0]*std_landmark[1]);

                have_correct_meas = true;   //at least one measurement is correct!
            }
        }
        if(!have_correct_meas)    //unlucky particle
        {
            particles[i].weight = 0;
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    for(int i=0;i!=num_particles;i++)
    {
        weights[i] = particles[i].weight;
    }
    std::discrete_distribution<int> distribution(weights.begin(),weights.end());
    std::vector<Particle> particles2 = particles;   //copy not to overwrite the initial particles
    for(int i=0;i!=num_particles;i++)
    {
        int rand_num = distribution(generator);
        particles[i] = particles2[rand_num];
    }
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
