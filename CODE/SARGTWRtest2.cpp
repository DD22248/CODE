// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <vector>
#include <algorithm>
using namespace Rcpp;
using namespace Eigen;
using namespace std;
//[[Rcpp::export]]
Eigen::MatrixXd Aginv_fun (Eigen::MatrixXd bL) {
  JacobiSVD<MatrixXd> svd(bL, ComputeThinU | ComputeThinV);
  MatrixXd U = svd.matrixU();
  MatrixXd V = svd.matrixV();
  VectorXd S = svd.singularValues();
  const double tol = 1e-35;
  VectorXd::Index nS = S.size();
  VectorXd positive(nS);
  double threshold = max(tol * S[0], 0.0);
  for (int j = 0; j < nS; ++j) {
    positive[j] = (S[j] > threshold) ? 1 : 0;
  }
  MatrixXd A_ginv;
  if (positive.all()) {
    MatrixXd inv_S = S.array().pow(-1).matrix().asDiagonal();
    A_ginv = V * inv_S * U.transpose();
  } else if (!positive.any()) {
    A_ginv = MatrixXd::Zero(bL.cols(), bL.rows());
  } else {
    MatrixXd V_pos = MatrixXd::Zero(V.rows(), positive.sum());
    MatrixXd U_pos = MatrixXd::Zero(U.rows(), positive.sum());
    VectorXd S_pos = VectorXd::Zero(positive.sum());
    int col_index = 0;
    for (int j = 0; j < nS; ++j) {
      if (positive[j] == 1) {
        V_pos.col(col_index) = V.col(j);
        U_pos.col(col_index) = U.col(j);
        S_pos[col_index] = S[j];
        ++col_index;
      }
    }
    MatrixXd inv_S_pos = S_pos.array().pow(-1).matrix().asDiagonal();
    A_ginv = V_pos * inv_S_pos * U_pos.transpose();
  }
  return A_ginv;
}

//[[Rcpp::export]]
double CKAICSARGTWRGL_fun(Eigen::MatrixXd W, Eigen::MatrixXd Q, Eigen::MatrixXd X, Eigen::VectorXd Y, int k0, Eigen::MatrixXd ds, int Ht, Eigen::MatrixXd dt) {
  MatrixXd Ym = Y.matrix();
  int n = X.rows();
  int p0 = X.cols();
  MatrixXd SQ = MatrixXd::Zero(n, n);
  MatrixXd Wh = MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd ds_row = ds.row(i);
    std::vector<double> ds_vec(ds_row.data(), ds_row.data() + ds_row.size());
    std::sort(ds_vec.begin(), ds_vec.end());
    double ks = ds_vec[k0 - 1];
    ks += 0.00001;
    double kt = Ht;
    Eigen::VectorXd W_i(n);
    for (int j = 0; j < n; ++j) {
      double term1 = std::pow(ds(i, j) / ks, 2);
      double term2 = std::pow(dt(i, j) / kt, 2);
      W_i(j) = (term1 + term2 <= 1) ? (0.75 * (1 - term1 - term2)) : 0;
    }
    Wh.row(i) = W_i;
    MatrixXd pQ = Q.transpose()*W_i.asDiagonal();
    MatrixXd bQ = pQ*Q;
    MatrixXd A_ginv0 = Aginv_fun(bQ);
    MatrixXd c = A_ginv0*pQ;
    SQ.row(i) = Q.row(i)*c;
  }
  Eigen::VectorXd yW_hat = SQ * W * Ym;
  Eigen::MatrixXd SL = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd ZG(n, p0 + 1);
  ZG << yW_hat, X;
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd W_i1 = Wh.row(i);
    MatrixXd px = ZG.transpose()*W_i1.asDiagonal();
    MatrixXd bx = px*ZG;
    MatrixXd A_ginv1 = Aginv_fun(bx);
    MatrixXd c1 = A_ginv1*px;
    SL.row(i) = ZG.row(i)*c1;
  }
  VectorXd Yhat = SL*Ym;
  double RSS = pow(Ym.array()-Yhat.array(),2).sum();
  double trR1 = SL.trace();
  double AIC = log(RSS/n)+(n+trR1)/(n-2-trR1);
  return AIC;
}
//[[Rcpp::export]]
Eigen::VectorXd CKAICSARGTWRGL_bT(Eigen::MatrixXd W, Eigen::MatrixXd Q, Eigen::MatrixXd X, Eigen::VectorXd Y, Eigen::VectorXi K0, Eigen::MatrixXd ds, Eigen::VectorXi Ht, Eigen::MatrixXd dt) {
  int nk = K0.size();
  int nt = Ht.size();
  VectorXd AIC_T = VectorXd::Zero(nt);
  VectorXd k0_T = VectorXd::Zero(nt);
  for (int j = 0; j < nt; ++j) {
    int t0 = Ht[j];
    int xL = K0[0];
    int xU = K0[K0.size()-1];
    const double gr = (sqrt(5) - 1) / 2;
    int  iter = 1;
    double d = gr * (xU - xL);
    int x1 = floor(xL + d);
    int x2 = round(xU - d);
    double f1 = CKAICSARGTWRGL_fun(W, Q, X, Y, x1, ds, t0, dt);
    double f2 = CKAICSARGTWRGL_fun(W, Q, X, Y, x2, ds, t0, dt);
    double d1 = f2 - f1;
    int xopt;
    double AICopt;
    if (f1 < f2) {
      xopt = x1;
      AICopt = f1;
    } else {
      xopt = x2;
      AICopt = f2;
    }
    while ((abs(d) > 1e-04) && (abs(d1) > 1e-04)) {
      d *= gr;
      if (f1 < f2) {
        xL = x2;
        x2 = x1;
        x1 = round(xL + d);
        f2 = f1;
        f1 = CKAICSARGTWRGL_fun(W, Q, X, Y, x1, ds, t0, dt);
      } else {
        xU = x1;
        x1 = x2;
        x2 = floor(xU - d);
        f1 = f2;
        f2 = CKAICSARGTWRGL_fun(W, Q, X, Y, x2, ds, t0, dt);
      }
      ++iter;
      if (f1 < f2) {
        xopt = x1;
        AICopt = f1;
      } else {
        xopt = x2;
        AICopt = f2;
      }
      d1 = f2 - f1;
    }
    AIC_T[j] = AICopt;
    k0_T[j] = xopt;
  }
  Index min_index;
  double min_value = AIC_T.minCoeff(&min_index);
  double k0opt = k0_T[min_index];
  double t0opt = Ht[min_index];
  VectorXd result(3);
  result(0) = min_value;
  result(1) = k0opt;
  result(2) = t0opt;
  return result;
}

//[[Rcpp::export]]
double CKAICSARGTWRLL_fun(Eigen::MatrixXd W, Eigen::MatrixXd Q, Eigen::VectorXd u, Eigen::VectorXd v, Eigen::VectorXd t, Eigen::MatrixXd X, Eigen::VectorXd Y, int k0, Eigen::MatrixXd ds, int Ht, Eigen::MatrixXd dt) {
  MatrixXd Ym = Y.matrix();
  int n = X.rows();
  int p0 = X.cols();
  MatrixXd SQ = MatrixXd::Zero(n, n);
  MatrixXd Wh = MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd ds_row = ds.row(i);
    std::vector<double> ds_vec(ds_row.data(), ds_row.data() + ds_row.size());
    std::sort(ds_vec.begin(), ds_vec.end());
    double ks = ds_vec[k0 - 1];
    ks += 0.00001;
    double kt = Ht;
    Eigen::VectorXd W_i(n);
    for (int j = 0; j < n; ++j) {
      double term1 = std::pow(ds(i, j) / ks, 2);
      double term2 = std::pow(dt(i, j) / kt, 2);
      W_i(j) = (term1 + term2 <= 1) ? (0.75 * (1 - term1 - term2)) : 0;
    }
    Wh.row(i) = W_i;
    MatrixXd pQ = Q.transpose()*W_i.asDiagonal();
    MatrixXd bQ = pQ*Q;
    MatrixXd A_ginv0 = Aginv_fun(bQ);
    MatrixXd c = A_ginv0*pQ;
    SQ.row(i) = Q.row(i)*c;
  }
  Eigen::VectorXd yW_hat = SQ * W * Ym;
  Eigen::MatrixXd SL = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd ZG(n, p0 + 1);
  ZG << yW_hat, X;
  int p = ZG.cols();
  MatrixXd Xnew = MatrixXd::Zero(n, 4*p);
  Xnew.block(0, 0, n, p) = ZG;
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd W_i1 = Wh.row(i);
    MatrixXd ZGUV = MatrixXd::Zero(n, 4*p);
    ZGUV.block(0, 0, n, p) = ZG;
    Eigen::VectorXd u_diff = u.array() - u(i);
    Eigen::VectorXd v_diff = v.array() - v(i);
    Eigen::VectorXd t_diff = t.array() - t(i);
    MatrixXd ZGu = ZG;
    MatrixXd ZGv = ZG;
    MatrixXd ZGt = ZG;
    for (int j = 0; j < p; ++j) {
      ZGu.col(j) = ZG.col(j).array() * u_diff.array();
      ZGv.col(j) = ZG.col(j).array() * v_diff.array();
      ZGt.col(j) = ZG.col(j).array() * t_diff.array();
    }
    ZGUV.block(0, p, n, p) = ZGu;
    ZGUV.block(0, p*2, n, p) = ZGv;
    ZGUV.block(0, p*3, n, p) = ZGt;
    MatrixXd pL = ZGUV.transpose()*W_i1.asDiagonal();
    MatrixXd bL = pL*ZGUV;
    MatrixXd A_ginv1 = Aginv_fun(bL);
    MatrixXd aL = A_ginv1*pL;
    SL.row(i) = Xnew.row(i)*aL;
  }
  VectorXd Yhat = SL*Ym;
  double RSS = pow(Ym.array()-Yhat.array(),2).sum();
  double trR1 = SL.trace();
  double AIC = log(RSS/n)+(n+trR1)/(n-2-trR1);
  return AIC;
}
//[[Rcpp::export]]
Eigen::VectorXd CKAICSARGTWRLL_bT(Eigen::MatrixXd W, Eigen::MatrixXd Q, Eigen::VectorXd u, Eigen::VectorXd v, Eigen::VectorXd t, Eigen::MatrixXd X, Eigen::VectorXd Y, Eigen::VectorXi K0, Eigen::MatrixXd ds, Eigen::VectorXi Ht, Eigen::MatrixXd dt) {
  int nk = K0.size();
  int nt = Ht.size();
  VectorXd AIC_T = VectorXd::Zero(nt);
  VectorXd k0_T = VectorXd::Zero(nt);
  for (int j = 0; j < nt; ++j) {
    int t0 = Ht[j];
    int xL = K0[0];
    int xU = K0[K0.size()-1];
    const double gr = (sqrt(5) - 1) / 2;
    int  iter = 1;
    double d = gr * (xU - xL);
    int x1 = floor(xL + d);
    int x2 = round(xU - d);
    double f1 = CKAICSARGTWRLL_fun(W,Q,u, v, t, X, Y, x1, ds, t0, dt);
    double f2 = CKAICSARGTWRLL_fun(W,Q,u, v, t, X, Y, x2, ds, t0, dt);
    double d1 = f2 - f1;
    int xopt;
    double AICopt;
    if (f1 < f2) {
      xopt = x1;
      AICopt = f1;
    } else {
      xopt = x2;
      AICopt = f2;
    }
    while ((abs(d) > 1e-04) && (abs(d1) > 1e-04)) {
      d *= gr;
      if (f1 < f2) {
        xL = x2;
        x2 = x1;
        x1 = round(xL + d);
        f2 = f1;
        f1 = CKAICSARGTWRLL_fun(W,Q,u, v, t, X, Y, x1, ds, t0, dt);
      } else {
        xU = x1;
        x1 = x2;
        x2 = floor(xU - d);
        f1 = f2;
        f2 = CKAICSARGTWRLL_fun(W,Q,u, v, t, X, Y, x2, ds, t0, dt);
      }
      ++iter;
      if (f1 < f2) {
        xopt = x1;
        AICopt = f1;
      } else {
        xopt = x2;
        AICopt = f2;
      }
      d1 = f2 - f1;
    }
    AIC_T[j] = AICopt;
    k0_T[j] = xopt;
  }
  Index min_index;
  double min_value = AIC_T.minCoeff(&min_index);
  double k0opt = k0_T[min_index];
  double t0opt = Ht[min_index];
  VectorXd result(3);
  result(0) = min_value;
  result(1) = k0opt;
  result(2) = t0opt;
  return result;
}

// [[Rcpp::export]]
Rcpp::List CGTWRLAR_H1(Eigen::MatrixXd W, Eigen::MatrixXd Q, Eigen::MatrixXd X, Eigen::VectorXd Y,
                       int k0, Eigen::MatrixXd ds, double ht, Eigen::MatrixXd dt) {
  
  MatrixXd Ym = Y.matrix();
  int n = X.rows();
  int p0 = X.cols();
  MatrixXd SQ = MatrixXd::Zero(n, n);
  MatrixXd Wh = MatrixXd::Zero(n, n);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd ds_row = ds.row(i);
    std::vector<double> ds_vec(ds_row.data(), ds_row.data() + ds_row.size());
    std::sort(ds_vec.begin(), ds_vec.end());
    double ks = ds_vec[k0 - 1];
    ks += 0.00001;
    double kt = ht;
    Eigen::VectorXd W_i(n);
    for (int j = 0; j < n; ++j) {
      double term1 = std::pow(ds(i, j) / ks, 2);
      double term2 = std::pow(dt(i, j) / kt, 2);
      W_i(j) = (term1 + term2 <= 1) ? (0.75 * (1 - term1 - term2)) : 0;
    }
    Wh.row(i) = W_i;
    MatrixXd pQ = Q.transpose()*W_i.asDiagonal();
    MatrixXd bQ = pQ*Q;
    MatrixXd A_ginv0 = Aginv_fun(bQ);
    MatrixXd c = A_ginv0*pQ;
    SQ.row(i) = Q.row(i)*c;
  }
  Eigen::VectorXd yW_hat = SQ * W * Ym;
  Eigen::MatrixXd ZG(n, p0 + 1);
  ZG << yW_hat, X;
  Eigen::MatrixXd S_H1 = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd W_i1 = Wh.row(i);
    MatrixXd px = ZG.transpose()*W_i1.asDiagonal();
    MatrixXd bL = px*ZG;
    MatrixXd A_ginv1 = Aginv_fun(bL);
    MatrixXd c1 = A_ginv1*px;
    S_H1.row(i) = ZG.row(i)*c1;
  }
  Eigen::VectorXd ehat = (I - S_H1) * Ym;
  Eigen::MatrixXd R1 = (I - S_H1).transpose() * (I - S_H1);
  MatrixXd temp1 = Ym.transpose() * R1 * Ym;
  double RSSH1 = temp1.trace();
  return Rcpp::List::create(
    Rcpp::Named("ehat") = ehat,
    Rcpp::Named("R1") = R1,
    Rcpp::Named("yW_hat") = yW_hat,
    Rcpp::Named("RSSH1") = RSSH1
  );
}
// [[Rcpp::export]]
Rcpp::List CGTWRGAR_H0(Eigen::MatrixXd W, Eigen::MatrixXd Xc, Eigen::MatrixXd X, Eigen::VectorXd Y,
                       int k0, Eigen::MatrixXd ds, double ht, Eigen::MatrixXd dt) {
  
  MatrixXd Ym = Y.matrix();
  int n = X.rows();
  int p0 = X.cols();
  Eigen::MatrixXd S0 = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd ds_row = ds.row(i);
    std::vector<double> ds_vec(ds_row.data(), ds_row.data() + ds_row.size());
    std::sort(ds_vec.begin(), ds_vec.end());
    double ks = ds_vec[k0 - 1];
    ks += 0.00001;
    double kt = ht;
    Eigen::VectorXd W_i(n);
    for (int j = 0; j < n; ++j) {
      double term1 = std::pow(ds(i, j) / ks, 2);
      double term2 = std::pow(dt(i, j) / kt, 2);
      W_i(j) = (term1 + term2 <= 1) ? (0.75 * (1 - term1 - term2)) : 0;
    }
    Eigen::VectorXd W_i2 = W_i;
    Eigen::MatrixXd px0 = X.transpose() * W_i2.asDiagonal();
    MatrixXd bx0 = px0*X;
    MatrixXd A_ginv2 = Aginv_fun(bx0);
    MatrixXd cc = A_ginv2*px0;
    S0.row(i) = X.row(i) * cc;
  }
  Eigen::MatrixXd px = Xc.transpose() * (I - S0).transpose() * (I - S0);
  MatrixXd bx = px*Xc;
  MatrixXd A_ginv0 = Aginv_fun(bx);
  MatrixXd cz = A_ginv0*px;
  Eigen::MatrixXd S_H0 = S0 + (I - S0) * Xc * cz;
  Eigen::MatrixXd R0 = (I - S_H0).transpose() * (I - S_H0);
  MatrixXd temp2 = Ym.transpose() * R0 * Ym;
  double RSSH0 = temp2.trace();
  Eigen::VectorXd bc = cz * Ym;
  Eigen::MatrixXd rhohat = bc.replicate(1, n).asDiagonal();
  Eigen::VectorXd MhatH0 = S0 * (I - rhohat * W) * Ym;
  return Rcpp::List::create(
    Rcpp::Named("rhohat") = rhohat,
    Rcpp::Named("MhatH0") = MhatH0,
    Rcpp::Named("R0") = R0,
    Rcpp::Named("RSSH0") = RSSH0
  );
}

// [[Rcpp::export]]
Rcpp::List CMixedGTWRLAR_H0(Eigen::MatrixXd W, Eigen::MatrixXd Xc, Eigen::MatrixXd Xv,
                            Eigen::MatrixXd yW_hat, Eigen::VectorXd Y,
                            int k0, Eigen::MatrixXd ds, double ht, Eigen::MatrixXd dt) {
  MatrixXd Ym = Y.matrix();
  int n = Xv.rows();
  Eigen::MatrixXd Zv(n, Xv.cols() + 1);
  Zv << yW_hat, Xv;
  Eigen::MatrixXd Sv = Eigen::MatrixXd::Zero(n, n);
  MatrixXd Wh = MatrixXd::Zero(n, n);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd ds_row = ds.row(i);
    std::vector<double> ds_vec(ds_row.data(), ds_row.data() + ds_row.size());
    std::sort(ds_vec.begin(), ds_vec.end());
    double ks = ds_vec[k0 - 1];
    ks += 0.00001;
    double kt = ht;
    Eigen::VectorXd W_i(n);
    for (int j = 0; j < n; ++j) {
      double term1 = std::pow(ds(i, j) / ks, 2);
      double term2 = std::pow(dt(i, j) / kt, 2);
      W_i(j) = (term1 + term2 <= 1) ? (0.75 * (1 - term1 - term2)) : 0;
    }
    Wh.row(i) = W_i;
    Eigen::MatrixXd pv = Zv.transpose() * W_i.asDiagonal();
    MatrixXd bv = pv*Zv;
    MatrixXd A_ginv2 = Aginv_fun(bv);
    MatrixXd cv = A_ginv2*pv;
    Sv.row(i) = Zv.row(i)*cv;
  }
  Eigen::MatrixXd px1 = Xc.transpose() * (I - Sv).transpose() * (I - Sv);
  MatrixXd bx1 = px1*Xc;
  MatrixXd A_ginv3 = Aginv_fun(bx1);
  MatrixXd cz = A_ginv3*px1;
  Eigen::MatrixXd S_H0 = Sv + (I - Sv) * Xc * cz;
  Eigen::VectorXd bc = cz * Ym;
  Eigen::MatrixXd bv = Eigen::MatrixXd::Zero(n, Zv.cols());
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd W_i3 = Wh.row(i);
    Eigen::MatrixXd pxb = Zv.transpose() * W_i3.asDiagonal();
    MatrixXd bxb = pxb*Zv;
    MatrixXd A_ginv4 = Aginv_fun(bxb);
    MatrixXd cc = A_ginv4*pxb;
    bv.row(i) = (cc * (Ym - Xc * bc)).transpose();
  }
  Eigen::MatrixXd rhohat = bv.col(0).asDiagonal();
  Eigen::MatrixXd R0 = (I - S_H0).transpose() * (I - S_H0);
  MatrixXd temp4 = Ym.transpose() * R0 * Ym;
  double RSSH0 = temp4.trace();
  Eigen::VectorXd MhatH0 = S_H0 * (I - rhohat * W) * Ym;  
  return Rcpp::List::create(
    Rcpp::Named("rhohat") = rhohat,
    Rcpp::Named("MhatH0") = MhatH0,
    Rcpp::Named("R0") = R0,
    Rcpp::Named("RSSH0") = RSSH0
  );
}

// [[Rcpp::export]]
Rcpp::List CGTWRLARLL_H1(Eigen::MatrixXd W, Eigen::MatrixXd Q,
                         Eigen::VectorXd u, Eigen::VectorXd v, Eigen::VectorXd t,
                         Eigen::MatrixXd X, Eigen::VectorXd Y,
                         int k0, Eigen::MatrixXd ds, double ht, Eigen::MatrixXd dt) {
  MatrixXd Ym = Y.matrix();
  int n = X.rows();
  int p0 = X.cols();
  MatrixXd SQ = MatrixXd::Zero(n, n);
  MatrixXd Wh = MatrixXd::Zero(n, n);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd ds_row = ds.row(i);
    std::vector<double> ds_vec(ds_row.data(), ds_row.data() + ds_row.size());
    std::sort(ds_vec.begin(), ds_vec.end());
    double ks = ds_vec[k0 - 1];
    ks += 0.00001;
    double kt = ht;
    Eigen::VectorXd W_i(n);
    for (int j = 0; j < n; ++j) {
      double term1 = std::pow(ds(i, j) / ks, 2);
      double term2 = std::pow(dt(i, j) / kt, 2);
      W_i(j) = (term1 + term2 <= 1) ? (0.75 * (1 - term1 - term2)) : 0;
    }
    Wh.row(i) = W_i;
    MatrixXd pQ = Q.transpose()*W_i.asDiagonal();
    MatrixXd bQ = pQ*Q;
    MatrixXd A_ginv0 = Aginv_fun(bQ);
    MatrixXd c = A_ginv0*pQ;
    SQ.row(i) = Q.row(i)*c;
  }
  Eigen::VectorXd yW_hat = SQ * W * Ym;
  Eigen::MatrixXd ZG(n, p0 + 1);
  ZG << yW_hat, X;
  Eigen::MatrixXd S_H1 = Eigen::MatrixXd::Zero(n, n);
  int p = ZG.cols();
  Eigen::MatrixXd Xnew = Eigen::MatrixXd::Zero(n, 4 * p);
  Xnew.block(0, 0, n, p) = ZG;
  std::vector<Eigen::MatrixXd> ZGUV(n);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd W_i1 = Wh.row(i);
    ZGUV[i].resize(n, 4 * p);
    Eigen::VectorXd u_diff = u.array() - u(i);
    Eigen::VectorXd v_diff = v.array() - v(i);
    Eigen::VectorXd t_diff = t.array() - t(i);
    ZGUV[i] << ZG, 
               ZG.array().colwise() * u_diff.array(),
               ZG.array().colwise() * v_diff.array(),
               ZG.array().colwise() * t_diff.array();
    Eigen::MatrixXd pL1 = ZGUV[i].transpose() * W_i1.asDiagonal();
    Eigen::MatrixXd bL1 = pL1*ZGUV[i];
    Eigen::MatrixXd A_ginv1 = Aginv_fun(bL1);
    Eigen::MatrixXd aL1 = A_ginv1*pL1;
    S_H1.row(i) = Xnew.row(i) * aL1;
  }
  Eigen::VectorXd ehat = (I - S_H1) * Ym;
  Eigen::MatrixXd R1 = (I - S_H1).transpose() * (I - S_H1);
  MatrixXd temp5 = Ym.transpose() * R1 * Ym;
  double RSSH1 = temp5.trace();
  return Rcpp::List::create(
    Rcpp::Named("ehat") = ehat,
    Rcpp::Named("R1") = R1,
    Rcpp::Named("yW_hat") = yW_hat,
    Rcpp::Named("RSSH1") = RSSH1
  );
}

// [[Rcpp::export]]
Rcpp::List CGTWRGARLL_H0(Eigen::MatrixXd W, Eigen::VectorXd u, Eigen::VectorXd v, 
                         Eigen::VectorXd t, Eigen::MatrixXd Xc, Eigen::MatrixXd X, 
                         Eigen::VectorXd Y, int k0, Eigen::MatrixXd ds, double ht, Eigen::MatrixXd dt) {
  MatrixXd Ym = Y.matrix();
  int n = X.rows();
  int p0 = X.cols();
  Eigen::MatrixXd Xnew0 = Eigen::MatrixXd::Zero(n, 4 * p0);
  Xnew0.block(0, 0, n, p0) = X;
  Eigen::MatrixXd S0 = Eigen::MatrixXd::Zero(n, n);
  std::vector<Eigen::MatrixXd> XUV(n);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd ds_row = ds.row(i);
    std::vector<double> ds_vec(ds_row.data(), ds_row.data() + ds_row.size());
    std::sort(ds_vec.begin(), ds_vec.end());
    double ks = ds_vec[k0 - 1];
    ks += 0.00001;
    double kt = ht;
    Eigen::VectorXd W_i(n);
    for (int j = 0; j < n; ++j) {
      double term1 = std::pow(ds(i, j) / ks, 2);
      double term2 = std::pow(dt(i, j) / kt, 2);
      W_i(j) = (term1 + term2 <= 1) ? (0.75 * (1 - term1 - term2)) : 0;
    }
    XUV[i].resize(n, 4 * p0);
    Eigen::VectorXd u_diff = u.array() - u(i);
    Eigen::VectorXd v_diff = v.array() - v(i);
    Eigen::VectorXd t_diff = t.array() - t(i);
    XUV[i] << X,
              X.array().colwise() * u_diff.array(),
              X.array().colwise() * v_diff.array(),
              X.array().colwise() * t_diff.array();
    Eigen::MatrixXd pL2 = XUV[i].transpose() * W_i.asDiagonal();
    Eigen::MatrixXd bL2 = pL2*XUV[i];
    Eigen::MatrixXd A_ginv2 = Aginv_fun(bL2);
    Eigen::MatrixXd aL2 = A_ginv2*pL2;
    S0.row(i) = Xnew0.row(i) * aL2;
  }
  Eigen::MatrixXd px = Xc.transpose() * (I - S0).transpose() * (I - S0);
  MatrixXd bx = px*Xc;
  MatrixXd A_ginv3 = Aginv_fun(bx);
  MatrixXd cz = A_ginv3*px;
  Eigen::MatrixXd S_H0 = S0 + (I - S0) * Xc * cz;
  Eigen::MatrixXd R0 = (I - S_H0).transpose() * (I - S_H0);
  MatrixXd temp2 = Ym.transpose() * R0 * Ym;
  double RSSH0 = temp2.trace();
  Eigen::VectorXd bc = cz * Ym;
  Eigen::MatrixXd rhohat = bc.replicate(1, n).asDiagonal();
  Eigen::VectorXd MhatH0 = S0 * (I - rhohat * W) * Ym;
  return Rcpp::List::create(
    Rcpp::Named("rhohat") = rhohat,
    Rcpp::Named("MhatH0") = MhatH0,
    Rcpp::Named("R0") = R0,
    Rcpp::Named("RSSH0") = RSSH0
  );
}

// [[Rcpp::export]]
Rcpp::List CMixedGTWRLARLL_H0(Eigen::MatrixXd W, Eigen::VectorXd u, Eigen::VectorXd v, Eigen::VectorXd t,
                              Eigen::MatrixXd Xc, Eigen::MatrixXd Xv, Eigen::MatrixXd yW_hat,
                              Eigen::VectorXd Y, int k0,
                              Eigen::MatrixXd ds, double ht, Eigen::MatrixXd dt) {
  MatrixXd Ym = Y.matrix();
  int n = Xv.rows();
  Eigen::MatrixXd Zv(n, Xv.cols() + 1);
  Zv << yW_hat, Xv;
  int p1 = Zv.cols();
  Eigen::MatrixXd Xnew1 = Eigen::MatrixXd::Zero(n, 4 * p1);
  Xnew1.block(0, 0, n, p1) = Zv;
  Eigen::MatrixXd Sv = Eigen::MatrixXd::Zero(n, n);
  std::vector<Eigen::MatrixXd> ZvUV(n);
  MatrixXd Wh = MatrixXd::Zero(n, n);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd ds_row = ds.row(i);
    std::vector<double> ds_vec(ds_row.data(), ds_row.data() + ds_row.size());
    std::sort(ds_vec.begin(), ds_vec.end());
    double ks = ds_vec[k0 - 1];
    ks += 0.00001;
    double kt = ht;
    Eigen::VectorXd W_i(n);
    for (int j = 0; j < n; ++j) {
      double term1 = std::pow(ds(i, j) / ks, 2);
      double term2 = std::pow(dt(i, j) / kt, 2);
      W_i(j) = (term1 + term2 <= 1) ? (0.75 * (1 - term1 - term2)) : 0;
    }
    Wh.row(i) = W_i;
    ZvUV[i].resize(n, 4 * p1);
    Eigen::VectorXd u_diff = u.array() - u(i);
    Eigen::VectorXd v_diff = v.array() - v(i);
    Eigen::VectorXd t_diff = t.array() - t(i);
    ZvUV[i] << Zv,
               Zv.array().colwise() * u_diff.array(),
               Zv.array().colwise() * v_diff.array(),
               Zv.array().colwise() * t_diff.array();
    Eigen::MatrixXd pL2 = ZvUV[i].transpose() * W_i.asDiagonal();
    Eigen::MatrixXd bL2 = pL2*ZvUV[i];
    Eigen::MatrixXd A_ginv2 = Aginv_fun(bL2);
    Eigen::MatrixXd aL2 = A_ginv2*pL2;
    Sv.row(i) = Xnew1.row(i) * aL2;
  }
  Eigen::MatrixXd px1 = Xc.transpose() * (I - Sv).transpose() * (I - Sv);
  MatrixXd bx1 = px1*Xc;
  MatrixXd A_ginv3 = Aginv_fun(bx1);
  MatrixXd cz = A_ginv3*px1;
  Eigen::MatrixXd S_H0 = Sv + (I - Sv) * Xc * cz;
  Eigen::VectorXd bc = cz * Ym;
  Eigen::MatrixXd Inew = Eigen::MatrixXd::Zero(p1, 4 * p1);
  Inew.block(0, 0, p1, p1) = Eigen::MatrixXd::Identity(p1, p1);
  Eigen::MatrixXd bv = Eigen::MatrixXd::Zero(n, p1);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd W_i3 = Wh.row(i);
    Eigen::MatrixXd pxb = ZvUV[i].transpose()*W_i3.asDiagonal();
    MatrixXd bxb = pxb*ZvUV[i];
    MatrixXd A_ginv4 = Aginv_fun(bxb);
    MatrixXd ccb = A_ginv4*pxb;
    bv.row(i) = (Inew * ccb * (Ym - Xc * bc)).transpose();
  }
  Eigen::MatrixXd rhohat = bv.col(0).asDiagonal();
  Eigen::MatrixXd R0 = (I - S_H0).transpose() * (I - S_H0);
  MatrixXd temp8 = Ym.transpose() * R0 * Ym;
  double RSSH0 = temp8.trace();
  Eigen::VectorXd MhatH0 = S_H0 * (I - rhohat * W) * Ym;  
  return Rcpp::List::create(
    Rcpp::Named("rhohat") = rhohat,
    Rcpp::Named("MhatH0") = MhatH0,
    Rcpp::Named("R0") = R0,
    Rcpp::Named("RSSH0") = RSSH0
  );
}
