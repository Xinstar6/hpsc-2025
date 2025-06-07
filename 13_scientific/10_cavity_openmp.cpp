#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace std;
typedef vector<vector<float>> matrix;

int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  matrix u(ny,vector<float>(nx));
  matrix v(ny,vector<float>(nx));
  matrix p(ny,vector<float>(nx));
  matrix b(ny,vector<float>(nx));
  matrix un(ny,vector<float>(nx));
  matrix vn(ny,vector<float>(nx));
  matrix pn(ny,vector<float>(nx));
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      u[j][i] = 0;
      v[j][i] = 0;
      p[j][i] = 0;
      b[j][i] = 0;
    }
  }

  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");

  for (int n=0; n<nt; n++) {
    #pragma omp parallel for collapse(2)
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        // Compute b[j][i]
        float dudx = (u[j][i + 1] - u[j][i - 1]) / (2.0f * dx);
        float dvdy = (v[j + 1][i] - v[j - 1][i]) / (2.0f * dy);
        float dudy = (u[j + 1][i] - u[j - 1][i]) / (2.0f * dy);
        float dvdx = (v[j][i + 1] - v[j][i - 1]) / (2.0f * dx);

        float divergence = dudx + dvdy;
        float nonlinear = dudx * dudx + 2.0f * dudy * dvdx + dvdy * dvdy;

        b[j][i] = rho * ((divergence / dt) - nonlinear);

      }
    }


    for (int it=0; it<nit; it++) {
      #pragma omp parallel for collapse(2)
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
	        pn[j][i] = p[j][i];

      #pragma omp parallel for collapse(2)
      for (int j=1; j<ny-1; j++) {
        for (int i=1; i<nx-1; i++) {
	      // Compute p[j][i]
          float weighted_x = dy * dy * (pn[j][i + 1] + pn[j][i - 1]);
          float weighted_y = dx * dx * (pn[j + 1][i] + pn[j - 1][i]);
          float rhs_term = b[j][i] * dx * dx * dy * dy;
          float denom_inv = 1.0f / (2.0f * (dx * dx + dy * dy));

          p[j][i] = (weighted_x + weighted_y - rhs_term) * denom_inv;
	      }
      }

      for (int j=0; j<ny; j++) {
        // Compute p[j][0] and p[j][nx-1]
        p[j][nx-1] = p[j][nx-2];
        p[j][0] = p[j][1];
      }
      for (int i=0; i<nx; i++) {
	    // Compute p[0][i] and p[ny-1][i]
        p[0][i] = p[1][i];
        p[ny-1][i] = 0;
      }
    }


    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        un[j][i] = u[j][i];
	      vn[j][i] = v[j][i];
      }
    }

    #pragma omp parallel for collapse(2)
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
	    // Compute u[j][i] and v[j][i]
        //----- update u[j][i] --------
        float adv_u_x = un[j][i] * (un[j][i] - un[j][i - 1]) / dx;
        float adv_u_y = un[j][i] * (un[j][i] - un[j - 1][i]) / dy;

        float gradp_u  = (p[j][i + 1] - p[j][i - 1]) / (2.0f * rho * dx);

        float lap_u_x  = (un[j][i + 1] - 2.0f * un[j][i] + un[j][i - 1]) / (dx * dx);
        float lap_u_y  = (un[j + 1][i] - 2.0f * un[j][i] + un[j - 1][i]) / (dy * dy);

        u[j][i] = un[j][i]
                  - dt * (adv_u_x + adv_u_y)
                  - dt * gradp_u
                  + nu * dt * (lap_u_x + lap_u_y);

        //----- update u[j][i] --------
        float adv_v_x = vn[j][i] * (vn[j][i] - vn[j][i - 1]) / dx;
        float adv_v_y = vn[j][i] * (vn[j][i] - vn[j - 1][i]) / dy;

        float gradp_v  = (p[j + 1][i] - p[j - 1][i]) / (2.0f * rho * dy);

        float lap_v_x  = (vn[j][i + 1] - 2.0f * vn[j][i] + vn[j][i - 1]) / (dx * dx);
        float lap_v_y  = (vn[j + 1][i] - 2.0f * vn[j][i] + vn[j - 1][i]) / (dy * dy);

        v[j][i] = vn[j][i]
                  - dt * (adv_v_x + adv_v_y)
                  - dt * gradp_v
                  + nu * dt * (lap_v_x + lap_v_y);
      }
    }

    for (int j=0; j<ny; j++) {
      // Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1]
      u[j][0] = 0;
      u[j][nx-1] = 0;
      v[j][0] = 0;
      v[j][nx-1] = 0;
    }

    for (int i=0; i<nx; i++) {
      // Compute u[0][i], u[ny-1][i], v[0][i], v[ny-1][i]
      u[0][i] = 0;
      u[ny-1][i] = 1;
      v[0][i] = 0;
      v[ny-1][i] = 0;
    }

    if (n % 10 == 0) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u[j][i] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v[j][i] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j][i] << " ";
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();
}
