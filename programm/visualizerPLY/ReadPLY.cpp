#include <vtkCamera.h>
#include <vtkDecimatePro.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkTriangleFilter.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkCleanPolyData.h>
#include <vtkPNGWriter.h>


int main(int argc, char* argv[])
{
  vtkSmartPointer<vtkPolyData> inputPolyData;
  if (argc > 1)
  {
    vtkNew<vtkSTLReader> reader;
    reader->SetFileName(argv[1]);
    vtkNew<vtkTriangleFilter> triangles;
    triangles->SetInputConnection(reader->GetOutputPort());
    triangles->Update();
    inputPolyData = triangles->GetOutput();
  }
  else
  {
    vtkNew<vtkSphereSource> sphereSource;
    sphereSource->SetThetaResolution(30);
    sphereSource->SetPhiResolution(15);
    sphereSource->Update();
    inputPolyData = sphereSource->GetOutput();
  }

  float reduction = 0.95; // 90% reduction
  if (argc > 2)
  {
    reduction = atof(argv[2]);
  }
  vtkNew<vtkNamedColors> colors;
  std::cout << "Before decimation" << std::endl << "------------" << std::endl;
  std::cout << "There are " << inputPolyData->GetNumberOfPoints() << " points."
            << std::endl;
  std::cout << "There are " << inputPolyData->GetNumberOfPolys() << " polygons."
            << std::endl;

  vtkNew<vtkDecimatePro> decimate;
  decimate->SetInputData(inputPolyData);
  decimate->SetTargetReduction(reduction);
  decimate->PreserveTopologyOff();
  decimate->Update();

  vtkNew<vtkPolyData> decimated;
  decimated->ShallowCopy(decimate->GetOutput());

  std::cout << "After decimation" << std::endl << "------------" << std::endl;

  std::cout << "There are " << decimated->GetNumberOfPoints() << " points."
            << std::endl;
  std::cout << "There are " << decimated->GetNumberOfPolys() << " polygons."
            << std::endl;
  std::cout << "Reduction: "
            << static_cast<double>((inputPolyData->GetNumberOfPolys() -
                                    decimated->GetNumberOfPolys())) /
          static_cast<double>(inputPolyData->GetNumberOfPolys())
            << std::endl;

  vtkNew<vtkPolyDataMapper> inputMapper;
  inputMapper->SetInputData(inputPolyData);

  vtkNew<vtkProperty> backFace;
  backFace->SetColor(colors->GetColor3d("Gold").GetData());

  vtkNew<vtkActor> inputActor;
  inputActor->SetMapper(inputMapper);
  inputActor->GetProperty()->SetInterpolationToFlat();
  inputActor->GetProperty()->SetColor(
      colors->GetColor3d("NavajoWhite").GetData());
  inputActor->SetBackfaceProperty(backFace);

  vtkNew<vtkPolyDataMapper> decimatedMapper;
  decimatedMapper->SetInputData(decimated);

  vtkNew<vtkActor> decimatedActor;
  decimatedActor->SetMapper(decimatedMapper);
  decimatedActor->GetProperty()->SetColor(
      colors->GetColor3d("NavajoWhite").GetData());
  decimatedActor->GetProperty()->SetInterpolationToFlat();
  decimatedActor->SetBackfaceProperty(backFace);

  // There will be one render window
  vtkNew<vtkRenderWindow> renderWindow;
  renderWindow->SetSize(600, 300);
  renderWindow->SetWindowName("Decimation");

  // And one interactor
  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(renderWindow);

  // Define viewport ranges
  // (xmin, ymin, xmax, ymax)
  double leftViewport[4] = {0.0, 0.0, 1.0, 1.0};
  double rightViewport[4] = {0.5, 0.0, 1.0, 1.0};

  // Setup both renderers
  vtkNew<vtkRenderer> leftRenderer;
  renderWindow->AddRenderer(leftRenderer);
  leftRenderer->SetViewport(leftViewport);
  leftRenderer->SetBackground(colors->GetColor3d("Peru").GetData());

  vtkNew<vtkRenderer> rightRenderer;
  //renderWindow->AddRenderer(rightRenderer);
  //rightRenderer->SetViewport(rightViewport);
  rightRenderer->SetBackground(colors->GetColor3d("CornflowerBlue").GetData());

  // Add the sphere to the left and the cube to the right
  leftRenderer->AddActor(inputActor);
  rightRenderer->AddActor(decimatedActor);

  // Shared camera
  // Shared camera looking down the -y axis
  vtkNew<vtkCamera> camera;
  camera->SetPosition(0, -1, 0);
  camera->SetFocalPoint(0, 0, 0);
  camera->SetViewUp(0, 0, 1);
  camera->Elevation(30);
  camera->Azimuth(30);

  vtkNew<vtkPNGWriter> writer;
  writer->SetFileName("screenshot2.png");
  writer->SetInputConnection(decimate->GetOutputPort());
  writer->Write();


  leftRenderer->SetActiveCamera(camera);
  rightRenderer->SetActiveCamera(camera);

  leftRenderer->ResetCamera();
  leftRenderer->ResetCameraClippingRange();

  renderWindow->Render();
  interactor->Start();

  return EXIT_SUCCESS;
}
