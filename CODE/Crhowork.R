Crhowork <- function(wdf1,c,m,T0,edis,clN){
  q <- 3
  n <- m^2*T0
  u <- c()
  v <- c()
  t <- c()
  beta <- matrix(0,n,q)
  rho1=rho2 <- c()
  for (i in 1:n){
    u[i] <- ((i-1)%%m)/(m-1)
    v[i] <- floor(((i-1)%%m^2)/m)/(m-1)
    t[i] <- floor((i-1)/m^2)+1
    rho1[i] <- 0.3+c*(v[i]^2-u[i]^2)*(t[i]/T0) 
    rho2[i] <- -0.3+c*sin(pi*(u[i]^2+v[i]^2+(t[i]/T0)^2))  
    beta[i,1] <- (u[i]+v[i])*exp(t[i]/10)
    beta[i,2] <- 32*u[i]*v[i]*(u[i]-1)*(v[i]-1)*t[i]*(t[i]-2)
    beta[i,3] <- -2-2*cos(pi*v[i]*exp(t[i]/4))     
  }
  data <- data.frame(u,v,t,rho1,rho2,beta)
  colnames(data) <- c("u","v","t","rho1","rho2","b1","b2","b3")
  library(limSolve)
  library(GWmodel)
  coords <- cbind(u,v)
  ds <- spDists(coords, coords, longlat = F)
  dt <- matrix(0,ncol=n,nrow=n)
  for(i in 1:n){
    for(j in 1:n){
      if(t[j]<=t[i]){
        dt[i,j] <- abs(t[i]-t[j])
      }else{
        dt[i,j] <- Inf
      }
    }
  }
 Wrho <- function(n,k,ds,dt){
    W <- matrix(0,n,n)
    for (i in 1:n) {
      dsk <- sort(ds[i,])[k]+0.00001
      ktt <- 3
      W[i,] <- ifelse(ds[i,]<=dsk & dt[i,] <= ktt,1,0)
      W[i,i] <- 0
    }
    return(W=W)
  }
  W0 <- as.matrix(Wrho(n,6,ds,dt))
  W1 <- rowSums(W0)
  W <- matrix(0,n,n)
  for (i in 1:n) {
    if(W1[i]!=0){
      W[i,] <- W0[i,]/W1[i]
    }
  }

  K0 <- seq(110,450,by=1)
  Ht <- c(1:T0)

  library(parallel)
  B <- 1000

  f <- function(N){
    source("testCPP.R")
    library(SARGTWRtest2)
    library(GWmodel)
    library(limSolve)
    seed <- 123456789+N
    set.seed(seed)
    X1 <- runif(n,-2,2)
    set.seed(seed+1)
    X2 <- rnorm(n,0,1)
    set.seed(seed+2)
    X3 <- rnorm(n,0,1)
    X <- cbind(X1,X2,X3)
    Xv <- cbind(X1,X2)
    Xc <- matrix(X3,nrow=n,ncol=1)
    Z1 <- W%*%X
    Z2 <- W%*%W%*%X
    Q <- cbind(X,Z1,Z2)
    set.seed(seed+3)
    if(edis==1){
      error = rnorm(n,0,1)
    }
    if(edis==2){
      error = runif(n,-sqrt(3),sqrt(3))
    }
    if(edis==3){
      error = rchisq(n,2)/2-1
    }
    rhouv1 <- diag(rho1)
    rhouv2 <- diag(rho2)
    Y1 <- (Solve(diag(n)-rhouv1%*%W,tol=1e-35))%*%(rowSums(X*beta)+error)
    Y2 <- (Solve(diag(n)-rhouv2%*%W,tol=1e-35))%*%(rowSums(X*beta)+error)

    Sys.setenv(USER_NAM="s")
    time1 <- system.time(
      outGL1 <- Crhotest(W,Q,X,Y1,K0,ds,Ht,dt,B)
    )
    time2 <- system.time(
      outLL1 <- CrhotestLL(W,Q,u,v,t,X,Y1,K0,ds,Ht,dt,B)
    )
    out1 <- cbind(outGL1,outLL1)
    outGL2 <- Crhotest(W,Q,X,Y2,K0,ds,Ht,dt,B)
    outLL2 <- CrhotestLL(W,Q,u,v,t,X,Y2,K0,ds,Ht,dt,B)
    out2 <- cbind(outGL2,outLL2)
    out <- cbind(out1,out2)
    out <- data.frame(out)
    names(out) <- c('M1G','M1LL','M2G','M2LL')
    time <- c(time1[3],time2[3])
    return(list(time=time,out=out))
  }
  clnum <- detectCores()-3
  cl <- makeCluster(getOption("cl.cores", clnum))
  system.time({
    res <- parLapply(cl,clN,f)
    stopCluster(cl)
  })
  return(res)
}

wdf1 <- "E:/result/rho"
c00 <- c(0,0.1,0.2,0.3,0.4,0.5)
cn <- length(c00)
N <- 200
test <- list()
Bp <- list(matrix(0,nrow = N,ncol = 4))
out <- list()
for (i in 1:cn){
  test[[i]] <- Crhowork(wdf1,c00[i],25,4,edis=1,1:N)
  Bp[[i]] <- test[[i]][[1]]$out[3,]
  for(j in 2:N){
    Bp[[i]] <- rbind(Bp[[i]],test[[i]][[j]]$out[3,])
  }
  out[[i]] <- matrix(c(apply(Bp[[i]],2,function(x)sum(x<0.01)),
                       apply(Bp[[i]],2,function(x)sum(x<0.05)),
                       apply(Bp[[i]],2,function(x)sum(x<0.10))),3,byrow = T)
}
c1 <- out[[1]]
c2 <- out[[2]]
c3 <- out[[3]]
c4 <- out[[4]]
c5 <- out[[5]]
c6 <- out[[6]]

Error1 <- array(c(c1,c2,c3,c4,c5,c6), dim = c(6, 3, 4)) 

dimnames(Error1)[[3]] <- c('c0','c.1','c.2','c.3','c.4','c.5')
for (mat_name in dimnames(Error1)[[3]]) {
  out <- data.frame(Error1[, , mat_name])
  names(out) <- c('M1G','M1LL','M2G','M2LL')
  write.csv(
    out,           
    file = paste0(mat_name, ".csv"),  
    row.names = FALSE                
  )
}
