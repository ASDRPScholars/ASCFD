from ascfd.simulation import is_inside_polygon

def test_point_inside_triangle():
    triangle = [(0, 0), (5, 0), (2.5, 5)]
    assert is_inside_polygon(triangle, (2.5, 2)) == True
    
def test_point_outside_triangle():
    triangle = [(0, 0), (5, 0), (2.5, 5)]
    assert is_inside_polygon(triangle, (5, 5)) == False
    
def test_left_edge_inside_square():
    square = [(0, 0), (0, 10), (10, 10), (10, 0)]
    assert is_inside_polygon(square, (0, 5)) == True
    
def test_right_edge_outside_square():
    square = [(0, 0), (0, 10), (10, 10), (10, 0)]
    assert is_inside_polygon(square, (10, 5)) == False
    
def test_corner_inside_square():
    square = [(0, 0), (0, 10), (10, 10), (10, 0)]
    assert is_inside_polygon(square, (0, 0)) == True
    
def test_point_inside_concave():
    concave = [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)]
    assert is_inside_polygon(concave, (2, 1)) == True
    
def test_interior_edge_inside_concave():
    concave = [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)]
    assert is_inside_polygon(concave, (3, 3)) == True