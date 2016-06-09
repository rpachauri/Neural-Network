import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

/**
 *This program is a model of a neural network.
 *This neural network will be used to tell if two bitmap files look similar.
 *
 *It works towards the best version of the neural network by adjusting the
 * weights using backwards propagation.
 *Back propagation is just an optimized version of steepest descent, and it
 * significantly reduces the amount of computation.
 *
 *I will be calling this a two-layer network since it has two layers of weights
 *
 *The Activation Function:
 *    f(x) = 1 / (1 + e^ (-x))
 *    
 *    this is what we use to calculate the dependent nodes:
 *       hidden nodes
 *       outputs
 *
 *Given the error function:
 *    E = 1/2 * ∑ (Tmi - Fmi)^2
 *    
 *    where
 *    Fmi is the Output Activation (what we got)
 *    Tmi is the Target Output Activation (what we want)
 *    m is the training-set index
 *    i is the target node index
 *    E is the error function
 *    
 *    We multiply the summation by 1/2 to remove an unneeded factor of two
 *       from the derivative and we want to find the minimum with respect to
 *       all the weights w for all the layers across all the training set data
 *    
 *    We try to make E approach 0. To find a minimum we generally would take
 *       the gradient with respect to the weights and set it equal to zero and
 *       then solve for all the weights through the simultaneous equations.
 *       The problem is that there may not be a true zero. Thus, as long as we
 *       are under the threshold error, that will suffice.
 *       
 * We modify each weight by the negative derivative and put in a "learning
 *    factor," which I will call "lambda"
 *
 * Below is the notation and connectivity model. There will be multiple models
 *    because there are different training sets.
 *
 *    a1
 *          wkj
 *                h1
 *   .                  wji
 *               .            F1
 *   .           .            
 *               .            Fi
 *   .                  wji
 *                hj
 *          wkj
 *    ak
 *
 *The number of inputs, hidden nodes, and outputs can be changed because there
 * will be lots of them when we use this program.
 *
 *I got my information from Dr. Nelson, my teacher. Below is the documentation
 * he wrote for me, which I used as a reference:
 *
 *https://athena2.harker.org/pluginfile.php/108013/mod_resource/content/4/
 * Minimizing%20the%20Error%20Function.pdf
 *
 *@author RyanPachauri
 *@creation 10/1/14
 */
public class XOROptimized
{  
   /*
    * Below are the two ways to stop the training.
    * 
    * MAX_COUNT is the maximum number of iterations we will run the method evaluateAndImproveNetwork()
    *    We only have this as a check so that we don't get stuck in an infinite loop
    * OKAY_ERROR is the minimum error the network must achieve
    */
   final static Integer MAX_COUNT = 100000000;
   final static double OKAY_ERROR = 0.000000001;

   /*
    * learning factor - used to change the amount by which we will train the weights
    * 
    * Use this variable to change the amount the weights change by
    */
   final double LAMBDA = 0.1; 

   /*
    * Use these variables to set constraints on assigning random weights in the method
    *    assignWeightsRandomly()
    * 
    * The node value will be between:
    *    RANDOM_ADDER         and         (RANDOM_MULTIPLER - 1) + RANDOM_ADDER
    */
   final static Integer RANDOM_MULTIPLIER = 2;
   final static Integer RANDOM_ADDER = -1;

   final static String WEIGHTS_FILE_NAME = "weights.txt";

   private Scanner in;        //used to read in user input

   private int numInputs;     //the number of inputs in the perceptron
   private int numHiddens;    //the number of hidden layer nodes in the perceptron
   private int numOutputs;    //the number of outputs in the perceptron
   private int numModels;     //the number of models we want to use for different training sets
   private int numTestModels; //the number of models we test for once the network has been trained

   /*
    * 2-D arrays of both the inputs and the outputs
    * [training set][node]
    * 
    * We have to keep different nodes for different training sets because what
    *    we are given is different for each training set
    *    
    * Use the activation function to calculate each output node
    *    Fi = f(thetai)
    *    
    *    where:
    *       Fi is the output node
    *       f(...) is the activation function
    *       thetai is ∑j hj * wji (refer to it below)
    */
   private double[][] inputs;          //what we are given
   private double[][] outputs;         //the results we arrive at after evaluating the network
   private double[][] T;               //the results we want (target outputs)

   private double[][] testInputs;      //inputs used to test the network once it is done running

   /*
    * 1-D array of the hidden nodes
    * We don't need to have a 2-D array because the hidden nodes are dependent
    *    on inputs and weights and can be changed easily
    * 
    * The hidden layer is fully connected to both its input and output layer.
    * 
    * Use the activation function to calculate each node, hj
    *    hj = f(thetaj)
    *    
    *    where:
    *       hj is the hidden node
    *       f(...) is the activation function
    *       thetaj is ∑k ak * wkj
    */
   private double[] hidden;

   private double[][] weightskj;       //layer of weights between input layer and hidden layer
   private double[][] weightsji;       //layer of weights between hidden layer and output layer

   /*
    * Defining these collections makes coding the back propagation algorithm
    *    much simpler.
    *    
    *    I will not be creating an omegai array because I don't need to.
    *    However, I'll provide its reference so you know what it means:
    *       omegai = Ti - Fi
    * 
    * It's a lot easier looking at the documentation Dr. Nelson provided so
    *    that you can see the equations.
    * 
    * The weights on the right connectivity layer are independent of the
    *    weights in the left connectivity layer
    * 
    * We'll be modifying weights on the fly, so we don't need to make copies
    */
   public double[] psii;   // omegai * f ' (thetai) -  for updating the right-most set of weights
   public double[] thetai; // ∑j hj * wji              "                                            "
   public double[] thetaj; // ∑k ak * wkj            -  for updating the left-most set of weights
   public double[] omegaj; // ∑i psii * wji
   public double[] psij;   // omegaj * f ' (thetaj)

   /**
    * Default Constructor for the Perceptron
    * 
    * Initializes all the private instance variables
    * Gives values to weights, hidden layer nodes, and outputs.
    * Inputs must be given by the user either manually or from a file
    * 
    * OKAY_ERROR is given a default value. User can change it.
    */
   public XOROptimized()
   {
      this.in = new Scanner(System.in);

      this.assignInputsAndTargets();
      this.assignWeights();

      this.outputs = new double[this.numModels][this.numOutputs];
      this.hidden = new double[this.numHiddens];
      this.psii = new double[this.numOutputs];
      this.thetai = new double[this.numOutputs];
      this.thetaj = new double[this.numHiddens];
      this.omegaj = new double[this.numHiddens];
      this.psij = new double[this.numHiddens];
   }//public XOROptimized()

   /**
    * Prints out the values in a 2D array of doubles
    * 
    * May not see it anywhere else in the code
    *    Used only to test to see if methods are inputting correct values
    *    for 2D arrays
    * 
    * @param array   the 2D array of doubles we want to print out the values of
    */
   public void print2DDoubleArray(double[][] array)
   {
      for (int outer = 0; outer < array.length; outer++)       //loops through outer part of 2D array (columns)
      {
         for (int inner = 0; inner < array[0].length; inner++) //loops through inner part of 2D array (rows)
         {
            System.out.print(array[outer][inner] + "\t");      //prints out values with space between each
         }                                                     
         System.out.println();                                 //next line
      }                                                        //for (int outer = 0; outer < array.length; outer++)
      return;
   }//public void print2DDoubleArray(double[][] array)

   /**
    * Assigns the inputs in one of three ways:
    *    from a file
    *    using a pattern
    * 
    *****
    * When I say assign inputs and targets, I mean that we are assigning
    * the test inputs and test targets as well i.e.:
    *    inputs
    *    T
    *    testInputs
    *    testTargets
    *****
    *    
    * Will assign inputs in one of the above two ways, depending on user's answer
    * 
    * @Postcondition numModels, numInputs, numOutputs, numTestModels,
    *                inputs[][], T[][], testInputs[][], and testTargets[][] must be assigned
    */
   public void assignInputsAndTargets()
   {
      System.out.println("Indicate the numbered method you would like to assign inputs and targets:");
      System.out.println("(1)\tFrom images");
      System.out.println("(2)\tFrom given patterns");
      
      int methodNumber = in.nextInt();
      switch (methodNumber)
      {
      case 1:
         this.assignInputsAndTargetsFromImages();
         break;
      case 2:
         this.assignPatternedInputsAndTargets();
         break;
      default:
         System.out.println("You must choose one way to assign inputs and targets! Please restart the program.\n");
         break;
      }//switch (methodNumber)
   }//public void assignInputsAndTargets()

   /**
    * Assigns the inputs from a file
    *    File must be the .bmp file that DibDump can understand;
    *             
    * @Postcondition numModels, numInputs, numOutputs, numTestModels,
    *                inputs[][], T[][], testInputs[][], and testTargets[][] must be assigned
    *                   if user wants to assign this way
    */
   public void assignInputsAndTargetsFromImages()
   {
      String inFilePathCharacters = "src/Images/Characters/";
      File directoryOfCharacters = new File(inFilePathCharacters);
      String [] characterFiles = directoryOfCharacters.list();
      this.numModels = characterFiles.length - 1;
      
      String inFilePathTestCharacters = "src/Images/TestCharacters/";
      File directoryOfTests = new File(inFilePathTestCharacters);
      String [] testFiles = directoryOfTests.list();
      this.numTestModels = testFiles.length - 1;
      
      DibDump temp = new DibDump();
      temp.readInBMP(inFilePathCharacters + characterFiles[1]);
      double[] flattened = temp.flattenImageArray();
      
      this.numInputs = flattened.length;
      this.numOutputs = 1;
      this.inputs = new double[this.numModels][this.numInputs];
      this.outputs = new double[this.numModels][this.numOutputs];
      this.T = new double[this.numModels][this.numOutputs];
      
      this.testInputs = new double[this.numTestModels][this.numInputs];

      double space = (.9)/(this.numModels - 1);

      for (int m = 0; m < this.numModels; m++)
      {
         temp = new DibDump();
         temp.readInBMP(inFilePathCharacters + characterFiles[m+1]);
         flattened = temp.flattenImageArray();
         for (int k = 0; k < this.numInputs; k++)
         {
            this.inputs[m][k] = flattened[k];
         }
         this.T[m][0] = space*(m+1);
      }//for (int m = 0; m < this.numModels; m++)
      for (int m = 0; m < this.numTestModels; m++)
      {
         temp = new DibDump();
         temp.readInBMP(inFilePathTestCharacters + testFiles[m+1]);
         flattened = temp.flattenImageArray();
         for (int k = 0; k < this.numInputs; k++)
         {
            this.testInputs[m][k] = flattened[k];
         }
      }
      //this.print2DDoubleArray(this.inputs);
      return;
   }//public void assignInputsAndTargetsFromFile()

   /**
    * Assigns the inputs and targets based on PATTERNS, the static private instance variable
    *    in InputPatterns
    * Assigns the test inputs and test targets based on TEST_PATTERNS, the other static private instance variable
    *    in InputPatterns
    *    
    * @Postcondition numModels, numInputs, numOutputs, numTestModels,
    *                inputs[][], T[][], testInputs[][], and testTargets[][] must be assigned
    *                   if user wants to assign this way
    */
   public void assignPatternedInputsAndTargets()
   {
      this.numModels = InputPatterns.PATTERNS.length;                         //sets numModels
      this.numInputs = InputPatterns.PATTERNS[0].getInputs().length;          //sets numInputs
      this.numOutputs = InputPatterns.PATTERNS[0].getExpectedOutputs().length;//sets numOutputs
      this.inputs = new double[this.numModels][this.numInputs];               //initializes inputs
      this.T = new double[this.numModels][this.numOutputs];                   //initializes targets
      for (int m = 0; m < this.numModels; m++)                                //loops through each model for inputs and targets
      {
         inputs[m] = InputPatterns.PATTERNS[m].getInputs();                   //sets inputs[m] to pattern's inputs
         T[m] = InputPatterns.PATTERNS[m].getExpectedOutputs();               //sets T[m] to pattern's expected outputs
      }

      this.numTestModels = InputPatterns.TEST_PATTERNS.length;
      this.testInputs = new double[this.numTestModels][this.numInputs];       //initializes testInputs
      for (int m = 0; m < this.numTestModels; m++)
      {
         this.testInputs[m] = InputPatterns.TEST_PATTERNS[m].getInputs();
      }
      return;
   }//public void assignPatternedInputsAndTargets()

   /**
    * Assigns the weights in one of two ways:
    *    randomly=
    *    from a file
    * Will assign weights in one of the above three ways, depending on user's answer
    * 
    * @Postcondition numHiddens, weightskj[][], weightsjk[][] must be assigned
    */
   public void assignWeights()
   {
      System.out.println("How would you like to assign weights:");
      System.out.println("(1)\tfromfile\n(2)\trandomly\n");

      int reply = in.nextInt();
      switch (reply)
      {
      case 1:
         this.assignWeightsFromFile();
         break;
      case 2:
         this.assignWeightsRandomly();
         break;
      default:
         System.out.println("You must choose a way to assign the weights."
               + "Please restart the program.");
         break;
      }//switch (reply)
      return;
   }//public void assignWeights()

   /**
    * Method assignWeightsFromFile() assigns the weights from a file
    * 
    * File is called whatever is saved to XOROptimized.WEIGHTS_FILE_NAME
    */
   public void assignWeightsFromFile()
   {
      try
      {
         System.out.println("How many hidden nodes were these inputs trained with?");
         this.numHiddens = in.nextInt();
         
         File file = new File(XOROptimized.WEIGHTS_FILE_NAME);
         Scanner inFile = new Scanner(file);
         
         /*
          * Creating a one-dimensional array of doubles to collect all the weights in the file
          *    Do this before setting private instance variables weightskj and weightsji
          *    It's easier to do this because there are no errors when using
          *       while (inFile.hasNextDouble()) but there are a couple of errors when using
          *       if (inFile.hasNextDouble())
          * It's worth not having to deal with the hassle of so many errors.
          *    After all, it's only an extra couple thousand calculations that we only do once.
          */
         double [] allWeights = new double[this.numInputs*this.numHiddens + this.numHiddens*this.numOutputs];
         int count = 0;
         while (inFile.hasNextDouble())
         {
            allWeights[count] = inFile.nextDouble();                       //collects the list of weights from file
            count++;
         }
         
         this.weightskj = new double[this.numInputs][this.numHiddens];     //initializes the weights
         this.weightsji = new double[this.numHiddens][this.numOutputs];
         
         /*
          * Sets the weightskj and weightsji using the collection of weights allWeights
          */
         for (int k = 0; k < this.numInputs; k++)
            for (int j = 0; j < this.numHiddens; j++)
               this.weightskj[k][j] = allWeights[k*this.numHiddens + j];
         
         for (int j = 0; j < this.numHiddens; j++)
            for (int i = 0; i < this.numOutputs; i++)
               this.weightsji[j][i] = allWeights[this.numInputs*this.numHiddens + j*this.numOutputs + i];
          
      }//try
      catch (FileNotFoundException e)
      {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
      return;
   }//public void assignWeightsFromFile()

   /**
    * Assigns random weights to each of the weights
    * 
    * The number will be between:
    *    RANDOM_ADDER         and         (RANDOM_MULTIPLER - 1) + RANDOM_ADDER
    *    
    * @Precondition  numInputs and numOutputs have been assigned
    * @Postcondition numHiddens, weightskj[][], weightsjk[][] must be assigned
    */
   public void assignWeightsRandomly()
   {
      System.out.println("How many hidden nodes should there be per model?");
      this.numHiddens = in.nextInt();
      this.weightskj = new double[this.numInputs][this.numHiddens];
      for (int k = 0; k < this.numInputs; k++)              //loops through outer loop of weightskj (size is numInputs)
         for (int j = 0; j < this.numHiddens; j++)          //loops through the inner loop of weightskj (size is numHiddens)
         {
            weightskj[k][j] = Math.random()*RANDOM_MULTIPLIER + RANDOM_ADDER;
         }
      this.weightsji = new double[this.numHiddens][this.numOutputs];
      for (int j = 0; j < this.numHiddens; j++)             //loops through outer loop of weightsji (size is numHiddens)
         for (int i = 0; i < this.numOutputs; i++)          //loops through inner loop of weightsji (size is numOutputs)
         {
            weightsji[j][i] = Math.random()*RANDOM_MULTIPLIER + RANDOM_ADDER;
         }
      return;
   }//public void assignRandomWeights()

   /**
    * This is our activation function. We use this to find out what the
    *    hidden nodes and outputs are.
    * 
    * ***If you change this, then you must change fPrime as well ***
    * 
    * f(x) = 1 / (1 + e^ (-x))
    * 
    * @param x    the double we are putting into the function
    * @return     the value when x is put into the function
    */
   public double f(double x)
   {
      return (1 / (1 + Math.exp(-x)));
   }

   /**
    * This is the derivative of the activation function. We will need it in calculating
    *    psiis and psijs
    * 
    * The derivative of f(x) = 1 / (1 + e^ (-x))
    *    is simply f'(x) = f(x) * (1 - f(x))
    * 
    * ***If you change the activation function, you must also change this method to match its derivative***
    * 
    * @param x    the double we are putting into the derivative of the activation function
    * @return     the value when x is put into the derivative of the activation function
    */
   public double fPrime(double x)
   {
      double fOfX = this.f(x);
      return fOfX * (1 - fOfX);
   }

   /**
    * Trains the network and improves the weights to minimize the error
    * 
    * Keeps improving the weights as long as the error is greater than the
    *    private instance variable OKAY_ERROR
    * Stops after
    *       MAX_COUNT + 1 iterations or
    *          MAX_COUNT + 1 because we evaluate and improve the network under each model once before the for loop starts
    *       the error is equal to or below what we want it to be (OKAY_ERROR)
    * And then prints out the error and the weights
    */
   public void train()
   {
      int count = 0;
      /*
       * check if the error is less than the okay error
       *    if it is, then we exit the for loop
       *    if it isn't, then we evaluate the entire network under each model
       *       and improve the weights
       * iterate MAX_COUNT times so we don't get caught in an infinite loop
       */
      while (this.evaluateAndImproveNetwork() > XOROptimized.OKAY_ERROR &&
            count < XOROptimized.MAX_COUNT)
      {
         if (count % 1000 == 0)
            System.out.println("Error"+ count + ":\t" + this.getError());//prints out the error for user
         count++;
      }
      System.out.println("Error:\t" + this.getError());      //prints out the error for user
      return;
   }//public void train()

   /**
    * We already have a set of inputs and targets that we have set aside for testing
    *    Resets inputs and T to match testInputs' and testTargets' dimensions
    * Resets everything dependent on numModels to have the proper dimensions
    *    inputs[][]
    *    T[][]
    *    outputs[][]
    * 
    * Changes the inputs and targets to the values in testInputs and testTargets
    * 
    * Tests the network against the test targets
    *    Supposed to be called once the weights are trained
    * 
    * Evaluates the network
    * Prints out the error and prints out the outputs
    * 
    * @Precondition  numTestModels, testInputs[][], and testTargets[][] have all been assigned
    */
   public void testTrain()
   {
      this.printEvaluatedOutputs();
      this.printEvaluatedTests();
      System.out.println("Would you like to save these weights? (yes / no)");
      String reply = in.next();
      if (reply.equals("yes"))
         this.saveWeights();
      return;
   }//public void testTrain()

   /**
    * Evaluates the network under all models
    *    Prints out the error of each model
    *    On the same line, prints out the outputs (in order)
    *    Moves to the next line so it can keep printing for the next model
    */
   public void printEvaluatedOutputs()
   {
      for (int m = 0; m < this.numModels; m++)//loops through each model
      {
         /*
          * Evaluates the network and prints out the error of the
          * network under that model
          */
         System.out.print("Error:\t" + this.evaluateNetwork(m));

         System.out.print("\t\tOutputs:");
         for (int i = 0; i < this.numOutputs; i++) {
            System.out.print("\t" + this.outputs[m][i]);
         }
         System.out.print("\t\tTargets:");
         for (int i = 0; i < this.numOutputs; i++) {
            System.out.print("\t" + this.T[m][i]);   
         }
         System.out.println();                                    
      }
      return;
   }//public void printEvaluatedOutputs()
   
   /**
    * Converts the inputs into the testInputs and evaluates the network
    */
   public void printEvaluatedTests()
   {
      this.inputs = new double[this.numTestModels][this.numInputs];           //resets inputs
      this.outputs = new double[this.numTestModels][this.numOutputs];         //resets outputs
      for (int m = 0; m < this.numTestModels; m++)
         for (int k = 0; k < this.numInputs; k++)
            this.inputs[m][k] = this.testInputs[m][k];                        //sets inputs equal to testInputs
      
      for (int m = 0; m < this.numTestModels; m++)
      {
         for (int i = 0; i < this.numOutputs; i++)                               //loops over the outputs
         {
            this.thetai[i] = 0;                                                  //resets thetai
            for (int j = 0; j < this.numHiddens; j++)                            //loops over the hidden nodes
            {
               this.thetaj[j] = 0;                                               //resets thetaj
               for (int k = 0; k < this.numInputs; k++)                          //loops over the inputs
               {
                  this.thetaj[j] += this.inputs[m][k] * this.weightskj[k][j];    //accumulates thetaj (sum of ak * wkj)
               }
               this.hidden[j] = this.f(this.thetaj[j]);                          //sets hidden node
               this.thetai[i] += this.hidden[j] * this.weightsji[j][i];          //accumulates thetai (sum of hj * wji)
            }//for (int j = 0; j < this.numHiddens; j++)

            this.outputs[m][i] = this.f(this.thetai[i]);                         //sets output
            System.out.println("Outputs:\t" + this.outputs[m][i]);               //prints outputs for user
         }//for (int i = 0; i < this.numOutputs; i++)
      }//for (int m = 0; m < this.numTestModels; m++)
   }//public void printEvaluatedTests()

   /**
    * Saves the weights
    */
   @SuppressWarnings("resource")
   public void saveWeights()
   {
      try
      {
         int count = 0;
         File file = new File(XOROptimized.WEIGHTS_FILE_NAME + "1");
         FileWriter fw = new FileWriter(file);
         fw.write(this.numHiddens);
         fw.flush();
         for (int k = 0; k < this.numInputs; k++)
         {
            for (int j = 0; j < this.numHiddens; j++)
            {
               fw.write("\n" + this.weightskj[k][j]);    //writes the weightskj onto the file
               count++;
            }
            fw.flush();
         }
         for (int j = 0; j < this.numHiddens; j++)
         {
            for (int i = 0; i < this.numOutputs; i++)
            {
               fw.write("\n" + this.weightsji[j][i]);    //writes the weightsji onto the file
               count++;
            }
            fw.flush();
         }
         System.out.println("Number of lines should be:\t" + count);
         System.out.println("Please remember that this program had "
               + this.hidden.length + " hidden nodes");
      }//try
      catch (IOException e)
      {
         e.printStackTrace();
      }
      return;
   }//public void saveWeights()

   /**
    * Prints out each of the weights, starting a new line for each one
    *    Different from print2DDoubleArray because it prints both sets
    *    of weights. Also shows the indexes of the weights
    *       e.g. w11, w211
    */
   public void printWeights()
   {
      for (int k = 0; k < this.numInputs; k++)                                      //loops through each input
         for (int j = 0; j < this.numHiddens; j++)                                  //loops through each node of hidden layer
         {
            System.out.println("w" + (k+1) + (j+1) + "1" + ":\t" + weightskj[k][j]);//prints out weights1
         }

      for (int j = 0; j < this.numHiddens; j++)                                     //loops through each node of hidden layer
         for (int i = 0; i < this.numOutputs; i++)                                  //loops through each output
         {
            System.out.println("w" + (j+1) + (i+1) + "2" + ":\t" + weightsji[j][i]);//prints out weights2
         }
      return;
   }//public void printWeights()

   /**
    * Evaluates the network within one model, only.
    * 
    * Accumulates the theta values and returns the subError
    * Recall the error function:
    * 
    *    E = 1/2 * ∑m,i (Tmi - Fmi)^2
    *    
    *    where
    *    Fmi is the Output Activation (what we got)
    *    Tmi is the Target Output Activation (what we want)
    *    m is the training-set index
    *    i is the target node index
    *    E is the error function
    *    
    *    This method will calculate the sum for one given m and all i
    * 
    * 
    * Below is the loop structure:
    * 
    *    subError = 0
    *       for i = 0 to Fi (output loop)
    *          thetai = 0
    *            
    *             for j = 0 to hj (hidden layer loop)
    *             
    *                if i = 0 (only need to do the following once for all outputs)
    *                   thetaj = 0
    *                   
    *                   for k = 0 to ak (input loop)
    *                     thetaj += ak * wkj (accumulates thetaj)
    *                   next k
    *               
    *                close if i = 0
    *                hj = f (thetaj)          (calculates hidden node)
    *                thetai += hj * wji       (accumulates thetai)
    *             next j
    *           
    *          Fi = f (thetai)                   (calculates output)
    *          omegai = (Ti - Fi)                (calculates omegai - no array)
    *          psii = omegai * f ' (thetai)      (calculates psii)
    *          subError += omegai * omegai       (adds to the error)
    *      
    *       next i
    *    return subError
    * 
    * @param model   the model we want to evaluate the network in
    * @return  double the (∑ (Tmi - Fmi) ^ 2) of this model
    */
   public double evaluateNetwork(int model)
   {  
      double subError = 0.0;
      for (int i = 0; i < this.numOutputs; i++)                               //loops over the outputs
      {
         this.thetai[i] = 0;                                                  //resets thetai
         for (int j = 0; j < this.numHiddens; j++)                            //loops over the hidden nodes
         {
            this.thetaj[j] = 0;                                               //resets thetaj
            for (int k = 0; k < this.numInputs; k++)                          //loops over the inputs
            {
               this.thetaj[j] += this.inputs[model][k] * this.weightskj[k][j];//accumulates thetaj (sum of ak * wkj)
            }
            this.hidden[j] = this.f(this.thetaj[j]);                          //sets hidden node
            this.thetai[i] += this.hidden[j] * this.weightsji[j][i];          //accumulates thetai (sum of hj * wji)
         }                                                                    //for (int j = 0; j < this.numHiddens; j++)

         this.outputs[model][i] = this.f(this.thetai[i]);                     //sets output
         double omegai = this.T[model][i] - this.outputs[model][i];           //creates omegai
         this.psii[i] = omegai * this.fPrime(this.thetai[i]);                 //sets psii
         subError += omegai * omegai;                                         //adds to subError
      }                                                                       //for (int i = 0; i < this.numOutputs; i++)
      return subError;
   }//public double evaluateNetwork(int model)

   /**
    * Adjusts the weights within one model
    *    Finds omegaj and psij to adjust weightsji
    * 
    *    for k = 0 to ak (input loop)
    *        
    *       for j = 0 to hj (hidden layer loop)
    *              
    *          if k = 0 (only need to do the following once for all inputs)
    *
    *             omegaj = 0
    *          
    *             for i = 0 to Fi (output loop)
    *                omegaj += psii weightsji (calculates psii * wji for omegaj)
    *                deltawji = lambda * hidden j * psii
    *                weightsji += deltawji
    *             next i
    *          
    *             psij = omegaj * f ' (thetaj)
    *          close if k = 0
    *       
    *          deltawkj = lambda * input k * psij
    *          weightskj += deltawkj
    *       
    *       next j
    *    next k
    * 
    * @precondition  thetaj, thetai, psii, omegai, inputs, hidden nodes, outputs
    *                   have been declared and are all under the same model
    * @param model   the model we want to adjust the weights in
    */
   public void improveWeights(int model)
   {
      for (int k = 0; k < this.numInputs; k++)                                   //loops over the inputs
      {
         for (int j = 0; j < this.numHiddens; j++)                               //loops over the hidden nodes
         {
            this.omegaj[j] = 0;                                                  //resets omegaj
            for (int i = 0; i < this.numOutputs; i++)                            //loops over the outputs
            {
               this.omegaj[j] += this.psii[i] * this.weightsji[j][i];            //accumulates omegaj
               double deltawji = this.LAMBDA * this.hidden[j] * this.psii[i];    //calculates change in weightsji
               this.weightsji[j][i] += deltawji;
            }
            this.psij[j] = omegaj[j] * this.fPrime(this.thetaj[j]);              //calculates psij
            double deltawkj = this.LAMBDA * this.inputs[model][k] * this.psij[j];//calculates change in weightskj
            this.weightskj[k][j] += deltawkj;
         }                                                                       //for (int j = 0; j < this.numHiddens; j++)
      }                                                                          //for (int k = 0; k < this.numInputs; k++)
      return;
   }//public void improveWeights(double model)

   /**
    * Changes the weights using backwards propagation as well as the hidden
    *    nodes and outputs.
    *    Collects theta values when the activation/weight multiplications
    *       are being performed when the network is being evaluated. Also need
    *       to maintain the activation values for the hidden layer
    * 
    * Recall the error function:
    * 
    *    E = 1/2 * ∑ (Tmi - Fmi)^2
    *    
    *    where
    *    Fmi is the Output Activation (what we got)
    *    Tmi is the Target Output Activation (what we want)
    *    m is the training-set index
    *    i is the target node index
    *    E is the error function
    * 
    * Loops over all the models
    *    Within the model loop, there are two sets of loops
    *
    * 1) 
    *    subError = 0
    *       for i = 0 to Fi (output loop)
    *          thetai = 0
    *            
    *             for j = 0 to hj (hidden layer loop)
    *             
    *                if i = 0 (only need to do the following once for all outputs)
    *                   thetaj = 0
    *                   
    *                   for k = 0 to ak (input loop)
    *                     thetaj += ak * wkj (accumulates thetaj)
    *                   next k
    *               
    *                close if i = 0
    *                hj = f (thetaj)          (calculates hidden node)
    *                thetai += hj * wji       (accumulates thetai)
    *             next j
    *           
    *          Fi = f (thetai)                   (calculates output)
    *          omegai = (Ti - Fi)                (calculates omegai - no array)
    *          psii = omegai * f ' (thetai)      (calculates psii)
    *          subError += omegai * omegai       (adds to the error)
    *      
    *       next i
    *    return subError
    *    
    * 2)
    *    for k = 0 to ak (input loop)
    *        
    *       for j = 0 to hj (hidden layer loop)
    *              
    *          if k = 0 (only need to do the following once for all inputs)
    *
    *             omegaj = 0
    *          
    *             for i = 0 to Fi (output loop)
    *                omegaj += psii weightsji (calculates psii * wji for omegaj)
    *                deltawji = lambda * hidden j * psii
    *                weightsji += deltawji
    *             next i
    *          
    *             psij = omegaj * f ' (thetaj)
    *          close if k = 0
    *       
    *          deltawkj = lambda * input k * psij
    *          weightskj += deltawkj
    *       
    *       next j
    *    next k
    *       
    * 
    * Then, at the very end, it returns 1/2 the error, which was the sum
    *    of all the (Tmi - Fmi)^2 over all models
    * 
    * @return  double   the error of the Network
    */
   public double evaluateAndImproveNetwork()
   {
      double error = 0;
      for (int m = 0; m < this.numModels; m++)//loops over all different models
      {
         error += this.evaluateNetwork(m);
         this.improveWeights(m);
      }
      return error/2;
   }//public double evaluateAndImproveNetwork()

   public double getError()
   {
      double error = 0;
      for (int m = 0; m < this.numModels; m++)//loops over all different models
         error += this.evaluateNetwork(m);
      return error/2;
   }//public double evaluateAndImproveNetwork()

   /**
    * Main method that gets called when we want to start the program
    * 
    * @param args
    */
   public static void main(String [] args)
   {
      XOROptimized perceptron = new XOROptimized();
      perceptron.train();
      perceptron.testTrain();
   }
}//public class XOROptimized