---
layout: page
title: ""
---

Summary posts on my university projects:

<ul>
{% for post in site.posts %}
    {% if post.categories contains "project" %}
        <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}
{% endfor %}
</ul>

