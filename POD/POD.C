#include "fvCFD.H"
#include "cellSet.H"
#include <Eigen/Core>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/SymEigsSolver.h>


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    IOdictionary PODDict
    (
        IOobject
        (
            "PODDict",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    );
    
    
    word fieldName(PODDict.lookup("fieldName"));
    label numberOfModes(readLabel(PODDict.lookup("numberOfModes")));
    word cellSetName(PODDict.lookupOrDefault<word>("cellSetName", ""));
    
    //word cellSetName(PODDict.lookup("cellSetName"));
    //const cellSet PODCellSet(mesh,cellSetName);
    //const labelList& PODCells = PODCellSet.toc();
    
    labelList PODCells; 
    label np;

    
    // Get time directories and matrix dimensions
    const instantList timeDirs = timeSelector::select0(runTime, args);
    label nt = timeDirs.size();
 

    if (cellSetName.empty())
    {
        // Use all cells in the mesh
        PODCells.resize(mesh.nCells());
        forAll(PODCells, i)
        {
            PODCells[i] = i;
        }
        np = mesh.nCells();
        Info << "No cellSet specified, performing POD on entire domain: " << np << " cells" << endl;
    }
    
    else
    {
        cellSet PODCellSet(mesh, cellSetName);
        labelList tempPODCells = PODCellSet.toc();
        PODCells.transfer(tempPODCells);
        np = PODCells.size();
        Info << "Performing POD on cellSet '" << cellSetName << "': " << np << " cells" << endl;
    }
    
    
    if (nt == 0 || np == 0)
    {
        FatalErrorInFunction
            << "No time directories or empty field found!" << exit(FatalError);
    }

    
    numberOfModes = min(numberOfModes, nt); // Ensure numberOfModes <= number of snapshots
    if (numberOfModes <= 0)
    {
        FatalErrorInFunction
            << "Invalid numberOfModes (" << numberOfModes << ")!" << exit(FatalError);
    }

    // Initialize snapshot matrix
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(np, nt);
    Info << "Snapshot matrix initialized: " << np << " x " << nt << endl;

    
    forAll(timeDirs, timeI)
    {
        runTime.setTime(timeDirs[timeI], timeI);
        Info << "Reading field at time = " << runTime.timeName() << endl;
        
        volScalarField PODField
        (
            IOobject
            (
                fieldName,
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        
        
        forAll(PODCells, i)
        {
            M(i, timeI) = PODField[PODCells[i]];
           
        }
    }

    
    Eigen::MatrixXd coMatrix = M.transpose() * M;
    Info << "Correlation matrix computed: " << nt << " x " << nt << endl;
    
    

    // Compute eigenvalues and eigenvectors using Spectra
    Spectra::DenseSymMatProd<double> op(coMatrix);
    label ncv = min(nt, max(2 * numberOfModes, numberOfModes + 10)); // Dynamic ncv
    Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, numberOfModes, ncv);
    
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestAlge);

    Eigen::VectorXd evalues;
    Eigen::MatrixXd evectors;
    
    if (eigs.info() == Spectra::CompInfo::Successful)
    {
        evalues = eigs.eigenvalues();
        evectors = eigs.eigenvectors();
        
        // Normalize eigenvalues
        evalues = evalues / evalues.sum();
    }
    else
    {
        FatalErrorInFunction
            << "Eigenvalue computation failed!" << exit(FatalError);
    }

    // Write eigenvalues
    {
        OFstream eigfile("Eigenvalues.dat");
        eigfile << "Eigenvalues" << nl;
        for (label i = 0; i < numberOfModes; ++i)
        {
            eigfile << evalues(i) << nl;
        }
    }
    Info << "Eigenvalues written to Eigenvalues.dat" << endl;

    // Write eigenvectors
    {
        OFstream vecfile("Eigenvectors.dat");
        vecfile << "Eigenvectors (" << evectors.rows() << " x " << numberOfModes << ")" << nl;
        for (Eigen::Index i = 0; i < evectors.rows(); ++i)
        {
            for (label j = 0; j < numberOfModes; ++j)
            {
                vecfile << evectors(i, j);
                if (j < numberOfModes - 1)
                {
                    vecfile << " ";
                }
            }
            vecfile << nl;
        }
    }
    Info << "Eigenvectors written to Eigenvectors.dat" << endl;    
    

    // Compute and write POD modes
    for (label modeI = 0; modeI < numberOfModes; ++modeI)
    {
        word modeName = "Mode" + Foam::name(modeI + 1);
        volScalarField mode
        (
            IOobject
            (
                modeName,
                runTime.constant(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(modeName, dimVelocity, Zero)
        );

        // Compute mode
        Eigen::VectorXd modeVec = M * evectors.col(modeI);
        modeVec.normalize();

        
        forAll(PODCells, i)
        {
            mode[PODCells[i]] = modeVec(i);
        }


        
        mode.write();
    }

    Info << numberOfModes << " POD modes written" << endl;
    Info << "Execution completed successfully" << endl;

    return 0;
}
