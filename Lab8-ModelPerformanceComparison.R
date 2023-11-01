# *****************************************************************************
# Lab 8.: Model Performance Comparison ----
#
# Course Code: BBT4206
# Course Name: Business Intelligence II
# Semester Duration: 21st August 2023 to 28th November 2023
#
# Lecturer: Allan Omondi
# Contact: aomondi [at] strathmore.edu
#
# Note: The lecture contains both theory and practice. This file forms part of
#       the practice. It has required lab work submissions that are graded for
#       coursework marks.
#
# License: GNU GPL-3.0-or-later
# See LICENSE file for licensing information.
# *****************************************************************************

# **[OPTIONAL] Initialization: Install and use renv ----
# The R Environment ("renv") package helps you create reproducible environments
# for your R projects. This is helpful when working in teams because it makes
# your R projects more isolated, portable and reproducible.

# Further reading:
#   Summary: https://rstudio.github.io/renv/
#   More detailed article: https://rstudio.github.io/renv/articles/renv.html

# "renv" It can be installed as follows:
# if (!is.element("renv", installed.packages()[, 1])) {
# install.packages("renv", dependencies = TRUE,
# repos = "https://cloud.r-project.org") # nolint
# }
# require("renv") # nolint

# Once installed, you can then use renv::init() to initialize renv in a new
# project.

# The prompt received after executing renv::init() is as shown below:
# This project already has a lockfile. What would you like to do?

# 1: Restore the project from the lockfile.
# 2: Discard the lockfile and re-initialize the project.
# 3: Activate the project without snapshotting or installing any packages.
# 4: Abort project initialization.

# Select option 1 to restore the project from the lockfile
# renv::init() # nolint

# This will set up a project library, containing all the packages you are
# currently using. The packages (and all the metadata needed to reinstall
# them) are recorded into a lockfile, renv.lock, and a .Rprofile ensures that
# the library is used every time you open the project.

# Consider a library as the location where packages are stored.
# Execute the following command to list all the libraries available in your
# computer:
.libPaths()

# One of the libraries should be a folder inside the project if you are using
# renv

# Then execute the following command to see which packages are available in
# each library:
lapply(.libPaths(), list.files)

# This can also be configured using the RStudio GUI when you click the project
# file, e.g., "BBT4206-R.Rproj" in the case of this project. Then
# navigate to the "Environments" tab and select "Use renv with this project".

# As you continue to work on your project, you can install and upgrade
# packages, using either:
# install.packages() and update.packages or
# renv::install() and renv::update()

# You can also clean up a project by removing unused packages using the
# following command: renv::clean()

# After you have confirmed that your code works as expected, use
# renv::snapshot(), AT THE END, to record the packages and their
# sources in the lockfile.

# Later, if you need to share your code with someone else or run your code on
# a new machine, your collaborator (or you) can call renv::restore() to
# reinstall the specific package versions recorded in the lockfile.

# [OPTIONAL]
# Execute the following code to reinstall the specific package versions
# recorded in the lockfile (restart R after executing the command):
# renv::restore() # nolint

# [OPTIONAL]
# If you get several errors setting up renv and you prefer not to use it, then
# you can deactivate it using the following command (restart R after executing
# the command):
# renv::deactivate() # nolint

# If renv::restore() did not install the "languageserver" package (required to
# use R for VS Code), then it can be installed manually as follows (restart R
# after executing the command):

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Introduction ----
# The performance of the trained models can be compared visually. This is done
# to help you to identify and choose the top performing models.

# STEP 1. Install and Load the Required Packages ----
## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## kernlab ----
if (require("kernlab")) {
  require("kernlab")
} else {
  install.packages("kernlab", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## randomForest ----
if (require("randomForest")) {
  require("randomForest")
} else {
  install.packages("randomForest", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# STEP 2. Load the Dataset ----
data(PimaIndiansDiabetes)

# STEP 3. The Resamples Function ----

# Analogy: We cannot compare apples with oranges; we compare apples with apples.

# The "resamples()"  function checks that the models are comparable and that
# they used the same training scheme ("trainControl" configuration).
# To do this, after the models are trained, they are added to a list and we
# pass this list of models as an argument to the resamples() function in R.

## 3.a. Train the Models ----
# We train the following models, all of which are using 10-fold repeated cross
# validation with 3 repeats:
#   LDA
#   CART
#   KNN
#   SVM
#   Random Forest

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

### LDA ----
set.seed(7)
diabetes_model_lda <- train(diabetes ~ ., data = PimaIndiansDiabetes,
                            method = "lda", trControl = train_control)

### CART ----
set.seed(7)
diabetes_model_cart <- train(diabetes ~ ., data = PimaIndiansDiabetes,
                             method = "rpart", trControl = train_control)

### KNN ----
set.seed(7)
diabetes_model_knn <- train(diabetes ~ ., data = PimaIndiansDiabetes,
                            method = "knn", trControl = train_control)

### SVM ----
set.seed(7)
diabetes_model_svm <- train(diabetes ~ ., data = PimaIndiansDiabetes,
                            method = "svmRadial", trControl = train_control)

### Random Forest ----
set.seed(7)
diabetes_model_rf <- train(diabetes ~ ., data = PimaIndiansDiabetes,
                           method = "rf", trControl = train_control)

## 3.b. Call the `resamples` Function ----
# We then create a list of the model results and pass the list as an argument
# to the `resamples` function.

results <- resamples(list(LDA = diabetes_model_lda, CART = diabetes_model_cart,
                          KNN = diabetes_model_knn, SVM = diabetes_model_svm,
                          RF = diabetes_model_rf))

# STEP 4. Display the Results ----
## 1. Table Summary ----
# This is the simplest comparison. It creates a table with one model per row
# and its corresponding evaluation metrics displayed per column.

summary(results)

## 2. Box and Whisker Plot ----
# This is useful for visually observing the spread of the estimated accuracies
# for different algorithms and how they relate.

scales <- list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(results, scales = scales)

## 3. Dot Plots ----
# They show both the mean estimated accuracy as well as the 95% confidence
# interval (e.g. the range in which 95% of observed scores fell).

scales <- list(x = list(relation = "free"), y = list(relation = "free"))
dotplot(results, scales = scales)

## 4. Scatter Plot Matrix ----
# This is useful when considering whether the predictions from two
# different algorithms are correlated. If weakly correlated, then they are good
# candidates for being combined in an ensemble prediction.

splom(results)

## 5. Pairwise xyPlots ----
# You can zoom in on one pairwise comparison of the accuracy of trial-folds for
# two models using an xyplot.

# xyplot plots to compare models
xyplot(results, models = c("LDA", "SVM"))

# or
# xyplot plots to compare models
xyplot(results, models = c("SVM", "CART"))

## 6. Statistical Significance Tests ----
# This is used to calculate the significance of the differences between the
# metric distributions of the various models.

### Upper Diagonal ----
# The upper diagonal of the table shows the estimated difference between the
# distributions. If we think that LDA is the most accurate model from looking
# at the previous graphs, we can get an estimate of how much better it is than
# specific other models in terms of absolute accuracy.

### Lower Diagonal ----
# The lower diagonal contains p-values of the null hypothesis.
# The null hypothesis is a claim that "the models are the same".
# A lower p-value is better (more significant).

diffs <- diff(results)

summary(diffs)

# **Required Lab Work Submission** ----
## Part A ----
# Create a new file called
# "Lab8-Submission-ModelPerformanceComparison.R".
# Provide all the code you have used to demonstrate the comparison of
# predictive models trained on a dataset other than the Pima Indians Diabetes
# Dataset.

## Part B ----
# Upload *the link* to your
# "Lab8-Submission-ModelPerformanceComparison.R" hosted
# on Github (do not upload the .R file itself) through the submission link
# provided on eLearning.

## Part C ----
# Create a markdown file called "Lab-Submission-Markdown.Rmd"
# and place it inside the folder called "markdown". Use R Studio to ensure the
# .Rmd file is based on the "GitHub Document (Markdown)" template when it is
# being created.

# Refer to the following file in Lab 1 for an example of a .Rmd file based on
# the "GitHub Document (Markdown)" template:
#     https://github.com/course-files/BBT4206-R-Lab1of15-LoadingDatasets/blob/main/markdown/BIProject-Template.Rmd # nolint

# Include Line 1 to 14 of BIProject-Template.Rmd in your .Rmd file to make it
# displayable on GitHub when rendered into its .md version

# It should have code chunks with explanations of all the steps performed.

## Part D ----
# Render the .Rmd (R markdown) file into its .md (markdown) version by using
# knitR in RStudio.

# You need to download and install "pandoc" to render the R markdown.
# Pandoc is a file converter that can be used to convert the following files:
#   https://pandoc.org/diagram.svgz?v=20230831075849

# Documentation:
#   https://pandoc.org/installing.html and
#   https://github.com/REditorSupport/vscode-R/wiki/R-Markdown

# By default, Rmd files are open as Markdown documents. To enable R Markdown
# features, you need to associate *.Rmd files with rmd language.
# Add an entry Item "*.Rmd" and Value "rmd" in the VS Code settings,
# "File Association" option.

# Documentation of knitR: https://www.rdocumentation.org/packages/knitr/

# Upload *the link* to "Lab-Submission-Markdown.md" (not .Rmd)
# markdown file hosted on Github (do not upload the .Rmd or .md markdown files)
# through the submission link provided on eLearning.