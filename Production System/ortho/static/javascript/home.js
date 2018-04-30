// dataArray = [20, 40, 50, 60]
//
// var width = 500
// var height = 500
//
// var widthScale = d3.scaleLinear()
//                 .domain([0,60])
//                 .range([0,width]);
//
// var color = d3.scaleLinear()
//             .domain([0,60])
//             .range(["red", "blue"]);
//
// var axis = d3.axisBottom()
//           .scale(widthScale)
//
// var canvas = d3.select("body")
//                 .append("svg")
//                 .attr("width", width)
//                 .attr("height", height)
//                 .append("g")
//                 .attr("transform", "translate(20,0)");
//
// var bars = canvas.selectAll("rect")
//                 .data(dataArray)
//                 .enter()
//                   .append("rect")
//                   .attr("width", function(d) { return widthScale(d); })
//                   .attr("height", 50)
//                   .attr("fill", function(d) { return color(d); } )
//                   .attr("y", function(d, i){return i*100;});
//
// canvas.append("g")
//     .attr("transform", "translate(0,400)")
//     .call(axis)


    // fake data
    var data = [
        {year: 2000, population: 1192},
        {year: 2001, population: 1234},
        {year: 2002, population: 1463},
        {year: 2003, population: 1537},
        {year: 2004, population: 1334},
        {year: 2005, population: 1134},
        {year: 2006, population: 1234},
        {year: 2007, population: 1484},
        {year: 2008, population: 1562},
        {year: 2009, population: 1427},
        {year: 2010, population: 1325},
        {year: 2011, population: 1484},
        {year: 2012, population: 1661},
        {year: 2013, population: 1537},
        {year: 2014, population: 1334},
        {year: 2015, population: 1134},
        {year: 2016, population: 1427}
    ];

    // Parse the date / time
    var parseDate = d3.timeParse("%Y");

    // force types
    function type(dataArray) {
        dataArray.forEach(function(d) {
            d.year = parseDate(d.year);
            d.retention = +d.population;
        });
        return dataArray;
    }
    data = type(data);

    // Set the dimensions of the canvas / graph
    var margin = {top: 30, right: 20, bottom: 30, left: 50},
        width = 500 - margin.left - margin.right,
        height = 250 - margin.top - margin.bottom;


    // Set the scales ranges
    var x = d3.scaleTime().range([0, width]);
    var y = d3.scaleLinear().range([height, 0]);

    // Define the axes
    var xAxis = d3.axisBottom().scale(x);
    var yAxis = d3.axisLeft().scale(y);

    // create a line based on the data
    var line = d3.line()
            .x(function(d) { return x(d.year); })
            .y(function(d) { return y(d.population); });

    // Add the svg canvas
    var svg = d3.select("body")
        .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
        .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // set the domain range from the data
    x.domain(d3.extent(data, function(d) { return d.year; }));
    y.domain([
            d3.min(data, function(d) { return Math.floor(d.population - 200); }),
            d3.max(data, function(d) { return Math.floor(d.population + 200); })
        ]);

    // draw the line created above
    svg.append("path").data([data])
            .style('fill', 'none')
            .style('stroke', 'steelblue')
            .style('stroke-width', '5px')
            .attr("d", line);

    // Add the X Axis
    svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis);

    // Add the Y Axis
    svg.append("g")
            .call(yAxis);
