Crhotest <- function(W,Q,X,Y,K0,ds,Ht,dt,B){
  n <- nrow(X)
  out_st <- CKAICSARGTWRGL_bT(W,Q,X,Y,K0,ds,Ht,dt)
  k0 <- out_st[2]
  ht <- out_st[3]
  #H1
  out_H1 <- CGTWRLAR_H1(W,Q,X,Y,k0,ds,ht,dt)
  ehat <- out_H1$ehat
  ecenter <- ehat-mean(ehat)
  RSSH1 <- out_H1$RSSH1
  R1 <- out_H1$R1
  Xc <- matrix(out_H1$yW_hat,nrow = n)
  #H0
  out_H0 <- CGTWRGAR_H0(W,Xc,X,Y,k0,ds,ht,dt)
  MhatH0 <- out_H0$MhatH0
  R0 <- out_H0$R0
  RSSH0 <- out_H0$RSSH0
  rhohat <- out_H0$rhohat
  #tvalue
  tvalue <- (RSSH0-RSSH1)/RSSH1
  Irho <- Solve(diag(n)-rhohat%*%W,tol=1e-55)
  #bootstrap
  theta <- function(ecenter){
    starY0 <- Irho%*%(MhatH0+ecenter) 
    num <- t(starY0)%*%(R0-R1)%*%starY0
    dec <- t(starY0)%*%R1%*%starY0
    startvalue <- num/dec
    out <- ifelse(startvalue>tvalue,1,0)
    return(out)
  }
  library(bootstrap)
  result <- bootstrap(ecenter,B,theta)
  B_pvalue <- mean(result$thetastar)
  result <- c(k0,ht,B_pvalue)
  return(result)
}
Cbetatest <- function(W,Q,Xc,Xv,X,Y,K0,ds,Ht,dt,B){
  n <- nrow(Xv)
  out_st <- CKAICSARGTWRGL_bT(W,Q,X,Y,K0,ds,Ht,dt)
  k0 <- out_st[2]
  ht <- out_st[3]
  #H1
  out_H1 <- CGTWRLAR_H1(W,Q,X,Y,k0,ds,ht,dt)
  ehat <- out_H1$ehat
  ecenter <- ehat-mean(ehat)
  RSSH1 <- out_H1$RSSH1
  R1 <- out_H1$R1
  yW_hat <- matrix(out_H1$yW_hat,nrow = n)
  #H0
  out_H0 <- CMixedGTWRLAR_H0(W,Xc,Xv,yW_hat,Y,k0,ds,ht,dt)
  MhatH0 <- out_H0$MhatH0
  R0 <- out_H0$R0
  RSSH0 <- out_H0$RSSH0
  rhohat <- out_H0$rhohat
  #tvalue
  tvalue <- (RSSH0-RSSH1)/RSSH1
  Irho <- Solve(diag(n)-rhohat%*%W,tol=1e-55)
  #bootstrap
  theta <- function(ecenter){
    starY0 <- Irho%*%(MhatH0+ecenter) # (solve(diag(n)-rhohat*W,tol=1e-55))%*%(rowSums(X*bv)+ecenter)
    num <- t(starY0)%*%(R0-R1)%*%starY0
    dec <- t(starY0)%*%R1%*%starY0
    startvalue <- num/dec
    out <- ifelse(startvalue>tvalue,1,0)
    return(out)
  }
  library(bootstrap)
  result <- bootstrap(ecenter,B,theta)
  B_pvalue <- mean(result$thetastar)
  result <- c(k0,ht,B_pvalue)
  return(result)
}

CrhotestLL <- function(W,Q,u,v,t,X,Y,K0,ds,Ht,dt,B){
  n <- nrow(X)
  out_st <- CKAICSARGTWRLL_bT(W,Q,u,v,t,X,Y,K0,ds,Ht,dt)
  k0 <- out_st[2]
  ht <- out_st[3]
  #H1
  out_H1 <- CGTWRLARLL_H1(W,Q,u,v,t,X,Y,k0,ds,ht,dt)
  ehat <- out_H1$ehat
  ecenter <- ehat-mean(ehat)
  RSSH1 <- out_H1$RSSH1
  R1 <- out_H1$R1
  Xc <- matrix(out_H1$yW_hat,nrow = n)
  #H0
  out_H0 <- CGTWRGARLL_H0(W,u,v,t,Xc,X,Y,k0,ds,ht,dt)
  MhatH0 <- out_H0$MhatH0
  R0 <- out_H0$R0
  RSSH0 <- out_H0$RSSH0
  rhohat <- out_H0$rhohat
  #tvalue
  tvalue <- (RSSH0-RSSH1)/RSSH1
  Irho <- Solve(diag(n)-rhohat%*%W,tol=1e-55)
  #bootstrap
  theta <- function(ecenter){
    starY0 <- Irho%*%(MhatH0+ecenter)
    num <- t(starY0)%*%(R0-R1)%*%starY0
    dec <- t(starY0)%*%R1%*%starY0
    startvalue <- num/dec
    out <- ifelse(startvalue>tvalue,1,0)
    return(out)
  }
  library(bootstrap)
  result <- bootstrap(ecenter,B,theta)
  B_pvalue <- mean(result$thetastar)
  result <- c(k0,ht,B_pvalue)
  return(result)
}
CbetatestLL <- function(W,Q,u,v,t,Xc,Xv,X,Y,K0,ds,Ht,dt,B){
  n <- nrow(Xv)
  out_st <- CKAICSARGTWRLL_bT(W,Q,u,v,t,X,Y,K0,ds,Ht,dt)
  k0 <- out_st[2]
  ht <- out_st[3]
  #H1
  out_H1 <- CGTWRLARLL_H1(W,Q,u,v,t,X,Y,k0,ds,ht,dt)
  ehat <- out_H1$ehat
  ecenter <- ehat-mean(ehat)
  RSSH1 <- out_H1$RSSH1
  R1 <- out_H1$R1
  yW_hat <- matrix(out_H1$yW_hat,nrow = n)
  #H0
  out_H0 <- CMixedGTWRLARLL_H0(W,u,v,t,Xc,Xv,yW_hat,Y,k0,ds,ht,dt)
  MhatH0 <- out_H0$MhatH0
  R0 <- out_H0$R0
  RSSH0 <- out_H0$RSSH0
  rhohat <- out_H0$rhohat
  #tvalue
  tvalue <- (RSSH0-RSSH1)/RSSH1
  Irho <- Solve(diag(n)-rhohat%*%W,tol=1e-55)
  #bootstrap
  theta <- function(ecenter){
    starY0 <- Irho%*%(MhatH0+ecenter) 
    num <- t(starY0)%*%(R0-R1)%*%starY0
    dec <- t(starY0)%*%R1%*%starY0
    startvalue <- num/dec
    out <- ifelse(startvalue>tvalue,1,0)
    return(out)
  }
  library(bootstrap)
  result <- bootstrap(ecenter,B,theta)
  B_pvalue <- mean(result$thetastar)
  result <- c(k0,ht,B_pvalue)
  return(result)
}