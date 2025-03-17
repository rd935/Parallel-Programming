#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

const double year = 365.25*24*60*60;
const float G = 6.67e-11;
/*
    Assume: all planets are in circular orbits
    Hardcode sun and earth
    in general: load in a file of positions

    Our solver uses simple high school math

    State of our system: Masses and positions and velocities

    f(m1,m2) = G*m1*m2/(d^2) 
    a(m1) = f(m1,m2)/(m1) = G*m2/(d^2)

    step forward in time: (Euler's method)
    v = v + a*dt
    x = x + v*dt

    Better: Runge-Kutta Fehlberg (RKF5)
*/
void add_body(vector<string> &names, vector<float> &Gm,
 vector<float> &x, vector<float> &y, vector<float> &z,
         vector<float> &vx, vector<float> &vy, vector<float> &vz,
         string name, float m, float x0, float y0, float z0, float vx0, float vy0, float vz0) {
    names.push_back(name);
    Gm.push_back(G * m);
    x.push_back(x0);
    y.push_back(y0);
    z.push_back(z0);
    vx.push_back(vx0);
    vy.push_back(vy0);
    vz.push_back(vz0);
}


// must pass by reference or original variables in main are unchanged
void initialize_solar_system(vector<string> &names,
 vector<float> &Gm, vector<float> &x, vector<float> &y, vector<float> &z,
  vector<float> &vx, vector<float> &vy, vector<float> &vz) {
    const float msun = 1.989e30f, mearth = 5.972e24f;
    add_body(names, Gm, x, y, z, vx, vy, vz, "Sun", msun, 0, 0, 0, 0, 0, 0);
    const float r = 149.6e9f;
    const double orbit_length = 2*M_PI*r; // distance around the sun
    const float v0 = orbit_length/year;
    add_body(names, Gm, x, y, z, vx, vy, vz, "Earth", mearth, r, 0, 0, 0, -v0, 0);
  }

  // Assuming that size(x) == size(y) == size(z) = size(vx) == size(vy) == size(vz)
void compute_acceleration(vector<float> &Gm, 
vector<float> &x, vector<float> &y, vector<float> &z,
vector<float> &vx, vector<float> &vy, vector<float> &vz,
vector<float> &ax, vector<float> &ay, vector<float> &az) {

    /*
        A: could assign 1 thread to each iteration of the loop

        B: could assign each thread a range of indices
        if you want to vectorize, you probably want choice B?    
    */
    #pragma omp parallel for
    for (int i = 0; i < x.size(); i++) {
        float x1 = x[i];
        float y1 = y[i];
        float z1 = z[i];
        float ax0 = 0;
        float ay0 = 0;
        float az0 = 0;
        for (int j = 0; j < x.size(); j++) {
            if (i == j) continue;
            float Gm2 = Gm[j];
            float x2 = x[j];
            float y2 = y[j];
            float z2 = z[j];
            float dx = x2 - x1;
            float dy = y2 - y1;
            float dz = z2 - z1;
            // calculate r^2 not r!
            float r2 = dx*dx + dy*dy + dz*dz;
            float r = sqrt(r2);
            float a_r = Gm2/r2/r;
            ax0 += a_r*dx;
            ay0 += a_r*dy;
            az0 += a_r*dz;
        }
        ax[i] = ax0;
        ay[i] = ay0;
        az[i] = az0;        
    }
}

// error: vx is not constant, can do a better approximation
void step_forward(vector<float> &x, vector<float> &y, vector<float> &z,
vector<float> &vx, vector<float> &vy, vector<float> &vz,
vector<float> &ax, vector<float> &ay, vector<float> &az, float dt) {
    for (int i = 0; i < x.size(); i++) {
        vx[i] += ax[i]*dt;
        vy[i] += ay[i]*dt;
        vz[i] += az[i]*dt;
        x[i] += vx[i]*dt;
        y[i] += vy[i]*dt;
        z[i] += vz[i]*dt;
    }
}

void print_system(const vector<string>& names,
const vector<float>& x, const vector<float>& y, const vector<float>& z,
const vector<float>& vx, const vector<float>& vy, const vector<float>& vz) {
    for (int i = 0; i < x.size(); i++) {
        cout << names[i] << " " << x[i] << " " << y[i] << " " << z[i] << " " << vx[i] << " " << vy[i] << " " << vz[i] << endl;
    }
}

int main() {
    vector<string> names;
    vector<float> Gm;
    vector<float> x, y, z; // should do each variable sequential, or block xyz together?
    vector<float> vx, vy, vz;
    vector<float> ax, ay, az;
    float dt = 100;
    const int num_steps = year/dt;
    initialize_solar_system(names, Gm, x, y, z, vx, vy, vz);
    for (int i = 0; i < num_steps; i++) {
        compute_acceleration(Gm, x, y, z, vx, vy, vz, ax, ay, az);
        step_forward(x, y, z, vx, vy, vz, ax, ay, az, dt);
    }
    print_system(names, x, y, z, vx, vy, vz);
}