
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .detect_pipe import remove_noise
from .project_algorithm import main
import logging
logging.basicConfig(filename='ImageProcessing.log',
                    level=logging.DEBUG, format='%(asctime)s:%(message)s')


@api_view(['POST'])
def pipe_caliber(request):
    if request.method == 'POST':
        image_root = request.data['Path']
        distance = opencv_pipe_detect(image_root)
        return Response({'Caliber_pixle': distance})


@api_view(['POST'])
def pipe_depth_cal(request):
    depth_list = list()
    if request.method == 'POST':
        img_root = request.data['Path']
        actual_external_edge_diameter = request.data['Edge']
        actual_external_pipe_diameter = request.data['Caliber']
        depth, pipe_depth, curve = remove_noise(
            img_root, actual_external_edge_diameter, actual_external_pipe_diameter)
        if depth < 0:
            depth = 0
        if pipe_depth < 0:
            pipe_depth = 0
        return Response(
            {
                'DepthToSurface': depth,
                'DepthToPipe': pipe_depth,
                'Type': curve[0],
                'Degree': curve[1],
            }
        )


@api_view(['POST'])
def pipe_assignment(request):
    if request.method == 'POST':
        image_root = request.data['Path']
        logging.debug('Path: {}'.format(image_root))
        try:
            depth, pipe_depth, degree, pipe_type = main(image_root)
        except:
            return Response({
                'DepthToPipe': 0,
                'PixelDistance': 0,
                'Type': 0,
                'Degree': 0
            })

        return Response(
            {
                'DepthToPipe': depth,
                'PixelDistance': pipe_depth,
                'Type': pipe_type,
                'Degree': degree
            }
        )
