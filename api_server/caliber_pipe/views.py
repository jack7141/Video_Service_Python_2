
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .detect_pipe import remove_noise

@api_view(['POST'])
def pipe_caliber(request):
    if request.method == 'POST':
        image_root = request.data['path']
        distance = opencv_pipe_detect(image_root)   
        return Response({'caliber_pixle': distance})

@api_view(['POST'])
def pipe_depth_cal(request):
    depth_list = list()
    if request.method == 'POST':
        img_root = request.data['path']
        actual_external_edge_diameter = request.data['edge']  
        actual_external_pipe_diameter = request.data['caliber']  
        depth, pipe_depth, curve = remove_noise(img_root,actual_external_edge_diameter,actual_external_pipe_diameter)
        if depth < 0:
            depth = 0
        if pipe_depth < 0:
            pipe_depth = 0 
        return Response(
            {
                'DepthToSurface': depth,
                'DepthToPipe' : pipe_depth,
                'Type': curve[0],
                'Degree': curve[1],
                }
            )
   