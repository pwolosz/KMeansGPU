library(ggplot2)

n <- 10000

# WELL DISTRIBUTED CLUSTERS
x1 <- 0
x2 <- 10
x3 <- 5
y1 <- 0
y2 <- 10
y3 <- 5
path <- 'D:\\Projects\\gpu\\KMeansGPU\\data1.txt'

x <- c()
y <- c()
z <- c()
x <- c(x,rnorm(n, x1, 1))
x <- c(x,rnorm(n, x2, 1))
x <- c(x,rnorm(n, x3, 1))

y <- c(y,rnorm(n, y1, 1))
y <- c(y,rnorm(n, y2, 1))
y <- c(y,rnorm(n, y3, 1))

z <- rep(1, length.out = 3*n)

data <- data.frame(x = x, y = y, z = z)

write.table(data, sep = ",", file = path, col.names = FALSE, row.names = FALSE)


# SINGLE LONG CLUSTER
x1 <- 5
y1 <- 5
std_x <- 1
std_y <- 5
x <- rnorm(3*n, x1, std_x)
y <- rnorm(3*n, y1, std_y)
z <- rep(1, length.out = 3*n)

data <- data.frame(x = x, y = y, z = z)

write.table(data, sep = ",", file = path, col.names = FALSE, row.names = FALSE)

# DRAW PLOT
in_path <- 'D:\\Projects\\gpu\\KMeansGPU\\out.txt'
in_data <- read.csv(in_path, sep = ",")

colnames(in_data) <- c("x", "y", "z", "label")

ggplot(in_data, aes(x = x, y = y, color = as.factor(label))) +
  geom_point() +
  theme(legend.position = "none")
